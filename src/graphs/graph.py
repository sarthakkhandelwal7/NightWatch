import os
import sys
from typing import Dict, List, Literal
import asyncio

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.tools import tool
from langgraph.types import Command, Send
from src.graphs.state import GraphState
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolNode
from pydantic import BaseModel, Field
from src.graphs.agents.orchestrator import OrchestratorAgent
from src.graphs.agents.investigator import InvestigationAgent
from src.graphs.agents.code_changes import CodeChangeAgent
from dotenv import load_dotenv

from langgraph.graph.message import add_messages

# Phoenix instrumentation setup
from openinference.instrumentation.langchain import LangChainInstrumentor
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk import trace as trace_sdk
from opentelemetry.sdk.trace.export import SimpleSpanProcessor

# Phoenix endpoint - using default port 6006
endpoint = "http://127.0.0.1:6006/v1/traces"
tracer_provider = trace_sdk.TracerProvider()
tracer_provider.add_span_processor(SimpleSpanProcessor(OTLPSpanExporter(endpoint)))

# Instrument LangChain for automatic tracing
LangChainInstrumentor().instrument(tracer_provider=tracer_provider)

class AgentGraph():
    def __init__(self) -> None:
        self.orchestrator_agent = OrchestratorAgent()
        self.investigator_agent = InvestigationAgent()
        self.code_change_agnet = CodeChangeAgent()
        self.graph = None
        
        
    async def get_orchestrator_node(self):
        async def orchestrator_node(state: GraphState) -> Command[Literal['tools', 'orchestrator_agent', 'investigation_agent']]:
            resp = await self.orchestrator_agent.llm_with_tools.ainvoke(state["orchestrator_agent_messages"]) 
            
            # Check if the response has tool calls (cast to AIMessage for type checking)
            
            if state['iteration_count_overall'] > 1 and hasattr(resp, 'tool_calls') and tool_calls[0]['name'] != 'create_jira_ticket':
                return Send(
                    node='tools',
                    arg={
                        'from': 'rate_limit',
                        'count': state['iteration_count_overall'],
                        "agent": "orchestrator_agent"
                    }
                )
            
            if hasattr(resp, 'tool_calls') and getattr(resp, 'tool_calls', None):
                tool_calls = getattr(resp, 'tool_calls')
                
                # Check if it's the Jira ticket creation (end of workflow)
                if len(tool_calls) == 1 and tool_calls[0]['name'] == 'create_jira_ticket':
                    # The LLM has decided to create Jira ticket - end the workflow
                    return Send(
                        node='tools',
                        arg={
                            'from': "orchestrator_agent",
                            "tool_calls": tool_calls
                        }
                    )
                    
                elif hasattr(resp, 'tool_calls') and resp.tool_calls and len(resp.tool_calls) > 1:
                    agents_to_call = []
                    agents_message_keys = {
                        "AnalysisInvestigationTool": ("investigation_agent", "investigation_agent_messages", 'investigation_agent_call_id'),
                        "ChangeInvestigatorTool": ("code_change_agent", "code_change_agent_messages", 'code_change_agent_call_id'), 
                        "HistoryInvestigationTool": ("historical_agent", "historical_agent_messages", 'orchestrator_agent_call_id')
                    }
                    update = {
                        "orchestrator_agent_messages": [resp],
                        "iteration_count_overall": state['iteration_count_overall']+1 
                        }
                    
                    for tool_call in tool_calls:
                        tool_name = tool_call['name']
                        if tool_name in agents_message_keys:
                            agent, message_key, id_key = agents_message_keys[tool_name]
                            agents_to_call.append(agent)  
                            update[id_key] = tool_call['id']
                            update[message_key] = [HumanMessage(content=tool_call['args']['message'], name="orchestrator_agent")]
                    return Command(
                        goto=agents_to_call,
                        update=update
                    )
            
            # If no valid tool calls, there's an issue with the prompt or LLM response
            return Command(
                update={
                    "orchestrator_agent_messages": [resp, HumanMessage(content="Please use the available tools to delegate investigation or create a Jira ticket.")],
                    "iteration_count_overall": state['iteration_count_overall']+1
                    },
                goto="orchestrator_agent"  # Stay in orchestrator to retry
            )
        
        return orchestrator_node

    async def get_investigation_node(self):
        """Returns the investigation agent node for the LangGraph workflow."""
        
        async def investigation_node(state: GraphState) -> Command[Literal['tools', 'orchestrator_agent', 'investigation_agent']]:
            """Investigation agent node that analyzes system metrics and logs."""
            
            resp = await self.investigator_agent.llm_with_tools.ainvoke(state["investigation_agent_messages"])
            # state["investigation_agent_messages"] = add_messages(state["investigation_agent_messages"], [resp])
            
            
            if state['iteration_count_investigator'] > 1 and hasattr(resp, 'tool_calls') and tool_calls[0]['name'] != 'OrchestratorStructuredResponseTool':
                return Send(
                    node='tools',
                    arg={
                        'from': 'rate_limit',
                        'count': state['iteration_count_investigator'],
                        "agent": "investigation_agent"
                    }
                )
                # return Command(
                #     update={
                #         "orchestrator_agent_messages": [
                #             HumanMessage(content="Agent exceeded maximum iterations - forcing report generation")
                #         ],
                #         "investigation_agent_messages": [
                #             SystemMessage(content="You must generate a final report now using OrchestratorStructuredResponseTool")
                #         ]
                #     },
                #     goto="investigation_agent"
                # )
            
            
            # Check if the agent used the OrchestratorStructuredResponseTool
            if hasattr(resp, 'tool_calls') and getattr(resp, 'tool_calls', None):
                tool_calls = getattr(resp, 'tool_calls')
                
                # Check if it's the structured response tool
                if len(tool_calls) == 1 and tool_calls[0]['name'] == 'OrchestratorStructuredResponseTool':
                    report = tool_calls[0]['args']['report']
                        
                    return Command(
                        update={
                            "orchestrator_agent_messages": [ToolMessage(content=report, name='investigation_agent', tool_call_id=state['investigation_agent_call_id'])],
                            "investigation_agent_messages": [resp],
                            "iteration_count_investigator": state['iteration_count_investigator'] + 1
                        },
                        goto='orchestrator_agent'
                    )
                else:
                    return Command(
                        update={
                            "investigation_agent_messages": [resp]
                        },
                        goto=Send('tools', arg={
                            "calls_made": state['iteration_count_investigator'],
                            "from": "investigation_agent", 
                            "tool_calls": tool_calls, 
                            
                            }
                        )
                    )
        
            
            # If no structured response tool was called, continue investigation
            return Command(
                update={"investigation_agent_messages": [resp, HumanMessage(content="No structured response tool was called, continue investigation")]},
                goto='investigation_agent'  # Stay in investigation for more tool calls
            )
        
        return investigation_node

    
    async def get_code_change_agent_node(self):
        async def code_change_agent_node(state: GraphState) -> Command[Literal['tools', 'orchestrator_agent', 'investigation_agent', END]]:
            resp = await self.code_change_agnet.llm_with_tools.ainvoke(state["code_change_agent_messages"])
            
            if state['iteration_count_code_change'] > 1 and hasattr(resp, 'tool_calls') and tool_calls[0]['name'] != 'OrchestratorStructuredResponseTool':
                return Send(
                    node='tools',
                    arg={
                        'from': 'rate_limit',
                        'count': state['iteration_count_code_change'],
                        "agent": "code_change_agent"
                    }
                )
                # return Command(
                #         update={
                #             "orchestrator_agent_messages": [
                #                 HumanMessage(content="Agent exceeded maximum iterations - forcing report generation")
                #             ],
                #             "code_change_agent_messages": [
                #                 SystemMessage(content="You must generate a final report now using OrchestratorStructuredResponseTool")
                #             ]
                #         },
                #         goto="code_change_agent"
                #     )
                        
            # Check if the agent used the OrchestratorStructuredResponseTool
            if hasattr(resp, 'tool_calls') and getattr(resp, 'tool_calls', None):
                tool_calls = getattr(resp, 'tool_calls')
                
                # Check if it's the structured response tool
                if len(tool_calls) == 1 and tool_calls[0]['name'] == 'OrchestratorStructuredResponseTool':
                    report = tool_calls[0]['args']['report']
                    
                    return Command(
                        update={
                            "orchestrator_agent_messages": [ToolMessage(content=report, name='code_change_agent', tool_call_id=state['code_change_agent_call_id'])],
                            "code_change_agent_messages": [resp],
                            "iteration_count_code_change": state['iteration_count_code_change'] + 1
                        },
                        goto='orchestrator_agent'
                    )
                else:
                    return Command(
                        update={
                            "code_change_agent_messages": [resp]
                        },
                        goto=Send('tools', arg={
                            "calls_made": state["iteration_count_code_change"],
                            "from": "code_change_agent", 
                            "tool_calls": tool_calls
                            }
                        )
                    )
        
            
            # If no structured response tool was called, continue code change investigation
            return Command(
                update={"code_change_agent_messages": [resp, HumanMessage(content="No structured response tool was called, continue investigation")]},
                goto='code_change_agent'  # Stay in code change agent for more tool calls
            )

        return code_change_agent_node
        
    async def get_tools_node(self):
        async def tools(state: Dict) -> Command[Literal['orchestrator_agent', 'investigation_agent', 'code_change_agent']]:
            try:
                
                if state['from'] == 'rate_limit':
                    print(f"Rate limit excdeed, number of iterations: {state['count']}")
                    return Command(
                        update={
                            "jira_ticket_id": f"Rate limit exceeded for {state['agent']} agent"
                        },
                        goto=END
                    )
                
                if state['from'] == 'orchestrator_agent':
                    import json
                    resp = json.loads((await self.orchestrator_agent.tools.ainvoke(state['tool_calls']))['messages'][0].content)
                    print(resp)
                    
                    ticket_id, report = resp
                    
                    return Command(
                        update={
                            "jira_ticket_id": ticket_id,
                            "report": report
                        },
                        goto=END
                    )
                
                elif state['from'] == 'investigation_agent':
                    investigation_agent_messages = (await self.investigator_agent.tools.ainvoke(state['tool_calls']))['messages'] + [HumanMessage(f'Number of tool calls made: {state['calls_made']+1}')]
                    # print("===================================")
                    # print("Print investigation agent messages")
                    # print(investigation_agent_messages)
                    # print(state['calls_made'])
                    # print("===================================")
                    return Command(
                        update={
                            "investigation_agent_messages": investigation_agent_messages,
                            'iteration_count_investigator': state['calls_made']+1
                        },
                        goto="investigation_agent"
                    )
                
                elif state['from'] == 'code_change_agent':
                    code_change_agent_messages = (await self.code_change_agnet.tools.ainvoke(state['tool_calls']))['messages'] + [HumanMessage(f'Number of tool calls made: {state['calls_made']+1}')]
                    # print("===================================")
                    # print("Print code change agent messages")
                    # print(code_change_agent_messages)
                    # print("===================================")
                    return Command(
                        update={
                            "code_change_agent_messages": code_change_agent_messages,
                            "iteration_count_code_change": state['calls_made']+1
                        },
                        goto="code_change_agent"
                    )
                else:
                    raise Exception('Unknown tool call in tools node')
            
            except Exception as e:
                print(f"Error in tools node: {e}")
                import traceback
                traceback.print_exc()
        
        return tools
                
                
                
    
    
    async def get_graph(self, state):
        self.graph = StateGraph(state)
        
        # Add nodes
        self.graph.add_node('orchestrator_agent', await self.get_orchestrator_node())
        self.graph.add_node('investigation_agent', await self.get_investigation_node())
        self.graph.add_node('code_change_agent', await self.get_code_change_agent_node())
        self.graph.add_node('tools', await self.get_tools_node())
        
        # Set entry point
        self.graph.set_entry_point('orchestrator_agent')
        
        # Add conditional edges based on agent decisions
        # The Command.goto will handle routing automatically
            
        return self.graph.compile()

async def main():
    agent = AgentGraph()
    
    # Create initial state with system prompts and incident context
    state = GraphState(
        incident_id="INC-2024-001",
        initial_alert={
            "function_name": "user-authentication-lambda",
            "error_type": "timeout",
            "timestamp": "2024-01-15T14:23:00Z",
            "region": "us-east-1"
        },
        incident_context={
            "function_name": "user-authentication-lambda",
            "error_type": "timeout",
            "timestamp": "2024-01-15T14:23:00Z",
            "region": "us-east-1",
            "error_count": 150,
            "duration": "30 seconds"
        },
        code_change_agent_messages=[SystemMessage(agent.code_change_agnet.system_prompt)],
        orchestrator_agent_messages=[
            SystemMessage(agent.orchestrator_agent.system_prompt),
            HumanMessage(content="New incident detected: Lambda function 'user-authentication-lambda' experiencing 150 timeout errors in the last 30 minutes. Please investigate and resolve.")
        ],
        investigation_agent_messages=[SystemMessage(agent.investigator_agent.system_prompt)],
        jira_ticket_id="",
        iteration_count_investigator=0,
        iteration_count_code_change=0,
        iteration_count_overall=0
    )
    
    print("🚀 Starting NightWatch Incident Response System...")
    print("📋 Incident: user-authentication-lambda timeout errors")
    print("⏰ Time: 2024-01-15T14:23:00Z")
    print("=" * 60)
    
    try:
        graph = await agent.get_graph(GraphState)
        print("✅ Graph compiled successfully")
        
        print("\n🔄 Executing incident response workflow...")
        resp = await graph.ainvoke(state)
        
        print("\n✅ Workflow completed successfully!")
        print("📊 Final State Summary:")
        print(f"   - Incident ID: {resp.get('incident_id', 'N/A')}")
        print(f"   - Jira Ticket: {resp.get('jira_ticket_id', 'N/A')}")
        print(f"   - Iterations: {resp.get('iteration_count', 0)}")
        
        # Show final messages from each agent
        if 'orchestrator_agent_messages' in resp:
            print(f"\n🎯 Orchestrator Final Message:")
            print(f"   {resp['orchestrator_agent_messages'][-1].content[:200]}...")
            
        if 'investigation_agent_messages' in resp:
            print(f"\n🔍 Investigation Agent Messages: {len(resp['investigation_agent_messages'])}")
            
        if 'code_change_agent_messages' in resp:
            print(f"\n📝 Code Change Agent Messages: {len(resp['code_change_agent_messages'])}")
            
        return resp
        
    except Exception as e:
        print(f"❌ Error during execution: {e}")
        import traceback
        traceback.print_exc()
        return None

 
if __name__ == '__main__':
    load_dotenv()
    # print(asyncio.run(main()))
    asyncio.run(main())