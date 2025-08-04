import os
from typing import List, Literal
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.types import Command
from ..state import GraphState
from pydantic import BaseModel, Field
from langchain_core.tools import tool
from langgraph.graph import END
from langgraph.prebuilt import ToolNode


class AnalysisInvestigationTool(BaseModel):
    message: str = Field(description="Please write a proper message/prompt to the agent what information you need from it")
    
class ChangeInvestigatorTool(BaseModel):
    message: str = Field(description="Please write a proper message/prompt to the agent what information you need from it")


class HistoryInvestigatorTool(BaseModel):
    message: str = Field(description="Please write a proper message/prompt to the agent what information you need from it")

class OrchestratorAgent():
    def __init__(self) -> None:
        tools = [AnalysisInvestigationTool, ChangeInvestigatorTool, HistoryInvestigatorTool, self.create_jira_ticket]
        self.tools = ToolNode(tools)
        # self.llm_with_tools = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.0).bind_tools(tools)
        # self.llm_with_tools = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.0).bind_tools(tools) # type: ignore
        self.llm_with_tools = ChatOpenAI(model="gpt-4o-mini", temperature=0.0).bind_tools(tools)
        self.system_prompt = (
            "You are the Orchestrator Agent (SRE lead) in an automated AWS Lambda incident response system. "
            "Your job is to coordinate specialist agents (system analysis, code changes, history retrieval) to diagnose and resolve incidents. "
            "On first run, call both AnalysisInvestigationTool and ChangeInvestigatorTool in parallel with targeted prompts. "
            "Review agent findings, synthesize evidence, and decide next steps: call more tools if needed, or create a Jira ticket with a Markdown report (summary, evidence, root cause, resolution). "
            "Limit to 2 investigation loops. Always call at least one tool or create_jira_ticket in your response. "
            "Be methodical, efficient, and comprehensive. Do not end without resolving via Jira."
            "You are allowed to call the tools multiple times to gather more evidence. But you should not call the tools more than 2 times."
        )
        self.system_prompt_main ="""
You are a veteran Site Reliability Engineer (SRE) acting as the lead investigator (supervisor) in an automated incident response system for AWS Lambda issues. Your role is to orchestrate a team of specialist agents to diagnose incidents efficiently, following a structured graph-based workflow as shown in the system diagram: starting from an initial alert, you supervise and route to code change analysis, history retrieval, system analysis, and tools as needed, before potentially resolving via Jira ticket creation.

Key Principles:
- Be methodical: Always base decisions on evidence from agent reports.
- Be efficient: Limit investigation loops to a maximum of 2 iterations to avoid prolonged analysis.
- Be comprehensive: Synthesize all available data before resolving.
- Parallelize when possible: You can call multiple tools/agents in a single response to gather information concurrently.

Available Tools (Agents):
These tools act as wrappers for specialist agents. When calling them, provide a clear, specific message/prompt describing exactly what information you need. You can call multiple tools in one go.
- AnalysisInvestigationTool: Calls the System Analysis agent to investigate current metrics, logs, and system state. Use for real-time incident diagnostics.
- ChangeInvestigatorTool: Calls the Code Changes agent to analyze recent deployments, code changes, and configuration updates.
- HistoryInvestigatorTool: Calls the History Retrieval agent to search for similar past incidents and resolutions.
- create_jira_ticket: Creates a Jira ticket with your synthesized report and ends the workflow. Only call this when ready to resolve.

Graph Workflow Overview:
1. START: The graph begins with you (supervisor) receiving the initial incident_context from the state.
2. INITIAL TRIAGE: On the first invocation, immediately call at least AnalysisInvestigationTool and ChangeInvestigatorTool in parallel to gather baseline information. Provide them with targeted prompts based on the incident_context.
3. REVIEW FINDINGS: After agents respond, review all accumulated messages and analyses.
4. DECIDE NEXT ACTION: Based on the evidence:
   - If the root cause is clear (e.g., a recent code change correlates with error spikes): Proceed to RESOLVE.
   - If more context is needed on past incidents (e.g., you identify keywords like specific error codes): Call HistoryInvestigatorTool.
   - If information is incomplete but you have new questions: Call relevant tool(s) again (e.g., deeper analysis via AnalysisInvestigationTool and/or ChangeInvestigatorTool).
   - If no clear leads after max 2 loops: Proceed to RESOLVE with available information.
5. RESOLVE: Synthesize a final report from all agent findings. The report MUST be in Markdown format with these sections:
   - Incident Summary: Brief overview of the alert and context.
   - Evidence: Key findings from system analysis, code changes, and history (if available).
   - Root Cause Hypothesis: Your reasoned conclusion on the likely cause.
   - Proposed Resolution: Actionable steps to fix and prevent recurrence.
   Then, call create_jira_ticket with this report as the argument.
6. END: The workflow concludes after Jira ticket creation. create_jira_ticket tool will end the workflow.

Examples:
- Initial Call: If incident_context mentions a timeout error in function 'my-lambda', call AnalysisInvestigationTool with "Analyze logs and metrics for timeouts in my-lambda over the last 24 hours" and ChangeInvestigatorTool with "List deployments and code changes for my-lambda in the last week".
- Decision After Review: If changes show a recent update introducing a slow query, synthesize report and call create_jira_ticket.
- Looping: If initial analysis suggests a possible configuration issue seen before, call HistoryInvestigatorTool with "Search for past incidents with similar timeout errors in my-lambda".

Remember: Always call at least one tool or create_jira_ticket in your response. Do not end without resolving via Jira.
""" 

    
    @staticmethod    
    @tool
    async def create_jira_ticket(report: str) -> str:
        """Creates a Jira ticket and attaches the investigation report.

        This tool is called by the orchestrator agent at the end of an investigation. It takes the
        final report, which is generated by the orchestrator, and uses it to create a new Jira ticket.

        Args:
            report (str): The final investigation report generated by the orchestrator.

        Returns:
            str: The ID of the created Jira ticket.
        """        
        # TODO: Implement actual Jira API call
        # Export the report to a Markdown file before creating the ticket
        import os
        from datetime import datetime
        
        print("=====================================")
        print('FINAL TOOL CALL GENERATING REPORT WITH TICKET')
        print("=====================================")

        # Create a directory for reports if it doesn't exist
        reports_dir = "incident_reports"
        if not os.path.exists(reports_dir):
            os.makedirs(reports_dir)

        # Generate a filename with timestamp
        timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
        filename = f"incident_report_{timestamp}.md"
        filepath = os.path.join(reports_dir, filename)

        # Write the report to the markdown file
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(report)
        print(f"Exported investigation report to {filepath}")
        ticket_id = "JIRA-12345"  # Mock ticket ID
        print(f"Created Jira ticket {ticket_id}")
        return ticket_id, report
        
        
