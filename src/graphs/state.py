from langgraph.graph.message import MessagesState, add_messages
from langchain_core.messages import AnyMessage

from typing import TypedDict, List, Dict, Any, Annotated

class GraphState(TypedDict):
    """
    Represents the state of our reactive graph for a Lambda incident.

    Attributes:
        incident_id: A unique ID for the incident.  
        initial_alert: The raw alert data from Amazon CloudWatch.
        incident_context: Parsed context (function_name, region, etc.).
        investigation_analysis: A correlated summary of log and metric findings.
        change_analysis: A summary of recent AWS Config and CodeDeploy changes.
        historical_analysis: A summary of relevant past incidents from Jira.
        synthesis_report: The final report with root cause and resolution.
        jira_ticket_id: The ID of the created Jira ticket.
        iteration_count: A counter to prevent infinite loops.
    """
    incident_id: str
    initial_alert: Dict[str, Any]
    incident_context: Dict[str, Any]
    investigation_agent_messages: Annotated[List[AnyMessage], add_messages]
    code_change_agent_messages: Annotated[List[AnyMessage], add_messages]
    historical_agent_messages: Annotated[List[AnyMessage], add_messages]
    orchestrator_agent_messages: Annotated[List[AnyMessage], add_messages]
    investigation_agent_call_id: str
    code_change_agent_call_id: str
    orchestrator_agent_call_id: str
    report: str
    jira_ticket_id: str
    iteration_count_overall: int
    iteration_count_investigator: int
    iteration_count_code_change: int
    iteration_count_historical: int
    
    
    
