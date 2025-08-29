from langgraph.graph import StateGraph
from .llm import LLMRegistry
from .retrievers.text2cypher_builder import Text2CypherRetrieverBuilder
from neo4j_graphrag.retrievers import Text2CypherRetriever
from .pydantictypes import AppState, Text2CypherRetrieverOutput, MultiTurnState
from neo4j_graphrag.embeddings.base import Embedder
from neo4j.exceptions import CypherSyntaxError
from neo4j_graphrag.exceptions import (
    Text2CypherRetrievalError,
    SearchValidationError,
)
from neo4j_graphrag.types import RetrieverResultItem
from typing import Optional, List, Dict, Any


class AgentService:
    def __init__(self, llm_registry: LLMRegistry, driver, database: str, embedder: Embedder):
        self.llm_registry = llm_registry
        self.driver = driver
        self.database = database
        self.text2cypher_retriever = self._build_text2cypher_retriever()
        self.graph = self._build_graph()
        self.embedder = embedder

    def _build_text2cypher_retriever(self) -> Text2CypherRetriever:
        return Text2CypherRetrieverBuilder(
            driver=self.driver,
            database=self.database,
            llm=self.llm_registry.langchain_llm
        ).build()

    def _text2cypher_node(self, state: MultiTurnState) -> MultiTurnState:
        """Generate Cypher query and execute it"""
        print(f"🔍 [text2cypher_node] Processing question: {state.current_question}")
        
        try:
            # Get search results from Text2Cypher retriever
            raw_result = self.text2cypher_retriever.get_search_results(state.current_question)
            print(f"🔍 [text2cypher_node] Raw result: {raw_result}")
            cypher_query = raw_result.metadata.get("cypher", "").strip()
            print(f"🔍 [text2cypher_node] Generated Cypher: {cypher_query}")

            # Format results
            formatter = self.text2cypher_retriever.result_formatter or (lambda r: RetrieverResultItem(content=str(r)))
            items = [formatter(r) for r in raw_result.records]
            
            # Update state with results
            state.results = Text2CypherRetrieverOutput(
                cypher=cypher_query,
                results=items
            )
            state.cypher_generated = cypher_query
            state.records_found = len(items)
            state.error_message = None
            
            print(f"✅ [text2cypher_node] Generated Cypher: {cypher_query}")
            print(f"✅ [text2cypher_node] Found {len(items)} records")
            
        except Text2CypherRetrievalError as e:
            print(f"❌ [text2cypher_node] Text2Cypher error: {e}")
            state.error_message = f"Failed to generate valid Cypher query: {str(e)}"
            state.needs_clarification = True
            state.clarification_request = "I couldn't understand your question well enough to generate a database query. Could you please provide more specific details about what you're looking for?"
            
        except CypherSyntaxError as e:
            print(f"❌ [text2cypher_node] Cypher syntax error: {e}")
            state.error_message = f"Generated Cypher had syntax error: {str(e)}"
            state.needs_clarification = True
            state.clarification_request = "I generated a query but it had a syntax error. Could you rephrase your question or provide more specific details?"
            
        except Exception as e:
            print(f"🔥 [text2cypher_node] Unexpected error: {e}")
            state.error_message = f"Unexpected error: {str(e)}"
            state.needs_clarification = True
            state.clarification_request = "I encountered an unexpected error. Could you try rephrasing your question?"
        
        return state

    def _evaluate_cypher_node(self, state: MultiTurnState) -> MultiTurnState:
        """Evaluate if the generated Cypher query matches the user's intent"""
        print(f"🔍 [evaluate_cypher_node] Evaluating Cypher: {state.cypher_generated}")
        
        if state.error_message:
            # If there was an error, we already set needs_clarification in text2cypher_node
            return state
        
        adapter = self.llm_registry.get_adapter("langgraph")
        
        # Create evaluation prompt
        evaluation_prompt = f"""
        You are an expert real estate data analyst evaluating if a Cypher query matches a user's question.
        
        User Question: "{state.current_question}"
        Generated Cypher Query: {state.cypher_generated}
        Number of Records Found: {state.records_found}
        
        Evaluate if the Cypher query:
        1. Accurately represents the user's intent
        2. Would return the type of data the user is asking for
        3. Uses appropriate filters and conditions
        
        Respond with ONLY one of these options:
        - "VALID" - if the query matches the user's intent
        - "NEEDS_CLARIFICATION" - if the query doesn't match or needs more specific details
        - "NO_RESULTS" - if the query is valid but returns no data (user should know this)
        
        If you choose "NEEDS_CLARIFICATION", also provide a brief explanation of what's missing or unclear.
        """
        
        try:
            evaluation_result = adapter.ask(evaluation_prompt).strip()
            print(f"🔍 [evaluate_cypher_node] Evaluation result: {evaluation_result}")
            
            if evaluation_result.startswith("NEEDS_CLARIFICATION"):
                state.needs_clarification = True
                # Extract clarification request if provided
                if ":" in evaluation_result:
                    state.clarification_request = evaluation_result.split(":", 1)[1].strip()
                else:
                    state.clarification_request = "I need more specific details to answer your question accurately. Could you provide more context?"
                    
            elif evaluation_result.startswith("NO_RESULTS"):
                state.needs_clarification = True
                state.clarification_request = "Your query is valid, but I found no matching data in the database. Would you like to try a broader search or provide different criteria?"
                
            else:  # VALID
                state.needs_clarification = False
                state.clarification_request = None
                
        except Exception as e:
            print(f"⚠️ [evaluate_cypher_node] Evaluation failed: {e}")
            # Default to valid if evaluation fails
            state.needs_clarification = False
            state.clarification_request = None
        
        return state

    def _format_response_node(self, state: MultiTurnState) -> MultiTurnState:
        """Format the final response for the user"""
        print(f"🔍 [format_response_node] Formatting response")
        
        if state.needs_clarification:
            # Return clarification request
            state.formatted_response = state.clarification_request
            state.llm_only_response = None
            return state
        
        adapter = self.llm_registry.get_adapter("langgraph")
        
        # Format successful response
        cypher = state.results.cypher if state.results else None
        records = state.results.results if state.results else []
        records_text = "\n".join(r.content for r in records)
        
        # LLM-only interpretation
        llm_only_prompt = f"""
        You are a CBRE real estate genai assistant. Interpret this question and give your best possible answer using your own knowledge.

        Question: "{state.current_question}"
        """.strip()

        try:
            state.llm_only_response = adapter.ask(llm_only_prompt)
        except Exception as e:
            state.llm_only_response = f"⚠️ Failed to generate LLM-only answer: {str(e)}"

        # Graph-enhanced response
        graph_prompt = f"""
        You are an expert real estate assistant. A Cypher query was run on a Neo4j database to answer this user question.

        Question: "{state.current_question}"

        Cypher Query: {cypher}

        Raw Results: {records_text}

        Please summarize the results in a clear, concise paragraph suitable for a non-technical audience. 
        If the data includes raw values or IDs, convert them into human-readable descriptions where possible. 
        Aim for a clean, informative summary that accurately conveys the core findings. If it is a list of results
        can you nicely format as a list output.
        """.strip()

        try:
            state.formatted_response = adapter.ask(graph_prompt)
        except Exception as e:
            state.formatted_response = f"⚠️ Failed to generate Graph-enhanced answer: {str(e)}"

        return state

    def _build_graph(self):
        """Build the simplified graph with only Text2Cypher and evaluation nodes"""
        builder = StateGraph(state_schema=MultiTurnState)

        # Add nodes
        builder.add_node("text2cypher", self._text2cypher_node)
        builder.add_node("evaluate", self._evaluate_cypher_node)
        builder.add_node("format", self._format_response_node)

        # Set entry point
        builder.set_entry_point("text2cypher")

        # Add edges
        builder.add_edge("text2cypher", "evaluate")
        builder.add_edge("evaluate", "format")

        # Set finish point
        builder.set_finish_point("format")

        return builder.compile()

    def run(self, question: str, conversation_history: Optional[List[Dict[str, Any]]] = None) -> MultiTurnState:
        """Run the multi-turn agent with conversation context"""
        # Initialize state with current question and conversation history
        initial_state = MultiTurnState(
            current_question=question,
            conversation_history=conversation_history or [],
            turn_number=len(conversation_history) + 1 if conversation_history else 1
        )
        
        # Run the graph
        raw_state = self.graph.invoke(initial_state)
        return MultiTurnState(**raw_state)

    def add_to_conversation(self, state: MultiTurnState, user_response: str) -> MultiTurnState:
        """Add user's clarification response to conversation and continue"""
        # Add the current interaction to conversation history
        interaction = {
            "question": state.current_question,
            "response": state.formatted_response or state.clarification_request,
            "needs_clarification": state.needs_clarification,
            "cypher": state.cypher_generated,
            "records_found": state.records_found
        }
        
        updated_history = state.conversation_history + [interaction]
        
        # Create new question combining original question with clarification
        combined_question = f"{state.current_question} Additional context: {user_response}"
        
        # Run the agent again with updated context
        return self.run(combined_question, updated_history)
