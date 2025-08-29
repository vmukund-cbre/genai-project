from neo4j import Driver, Record
from neo4j_graphrag.types import RetrieverResultItem
from neo4j_graphrag.schema import get_schema
from neo4j_graphrag.retrievers import Text2CypherRetriever
from neo4j_graphrag.llm import LLMInterface
from ..utils.query_examples import load_query_examples

PROMPT_TEMPLATE = """
You are a Cypher-generating expert for a CBRE real estate Neo4j graph.

Please closely align any generated cypher with properties in the schema.

Use the schema below to determine the correct traversal pattern.
Default to simple Cypher patterns such as:
  MATCH (a)-[:REL_TYPE]->(b)

Schema:
{schema}

Examples:
{examples}

User Question:
{query_text}

Instructions:
- Use exact labels and relationships from the schema.
- It is ok to use APOC.date 
- Return only a Cypher query. No markdown or commentary.
- Use `RETURN` statements that return only relevant properties.
- Focus on real estate data like properties, tenants, leases, financial metrics, etc.
- when you interact with the organization node in neo4j for country and customer use key instead of name
- try to be respectful of the property names of the schema when generating the cyper query
"""

class Text2CypherRetrieverBuilder:
    def __init__(self, driver: Driver, database: str, llm: LLMInterface, examples_file: str = "query_examples.yml"):
        self.driver = driver
        self.database = database
        self.llm = llm
        self.examples_file = examples_file

    def build(self) -> Text2CypherRetriever:
        schema = self._load_schema()
        prompt_template = self._build_prompt(schema)
        examples = self._get_examples()
        
        print(f"📊 Schema loaded: {len(schema)} characters")
        print(f"📝 Examples loaded: {len(examples)} examples")

        retriever = Text2CypherRetriever(
            driver=self.driver,
            llm=self.llm,
            neo4j_schema=schema,
            neo4j_database=self.database,
            examples=examples,
            custom_prompt=prompt_template,
            result_formatter=self._format_result
        )

        print("\n🔎 Text2CypherRetriever Summary")
        print("────────────────────────────────────────")
        print("🔤 LLM type:", type(retriever.llm))
        print("🧠 LLM model name:", getattr(retriever.llm, 'model_name', 'Unknown'))
        print(f"📁 Examples file: {self.examples_file}")
        print(f"📝 Examples count: {len(examples)}")
        print("\n📜 Neo4j Schema Snippet:")
        print(retriever.neo4j_schema[:1000], "...")
        print("\n🧪 Example Examples:")
        for example in examples[:3]:
            print("-", example[:100], "...")
        print("────────────────────────────────────────\n")

        return retriever

    def _load_schema(self) -> str:
        return get_schema(self.driver, database=self.database)

    def _build_prompt(self, schema: str) -> str:
        return PROMPT_TEMPLATE.format(
            schema=schema.replace("{", "[").replace("}", "]"),
            examples="{examples}",
            query_text="{query_text}"
        )

    def _get_examples(self) -> list[str]:
        """Load examples from external YAML file"""
        examples = load_query_examples(self.examples_file)
        
        # Fallback to default examples if YAML file is empty or not found
        if not examples:
            print("⚠️ No examples loaded from YAML, using fallback examples")
            examples = [
                "User: Show me all office properties in downtown areas\nCypher: MATCH (p:Property) WHERE p.property_type = 'Office' AND p.location CONTAINS 'downtown' RETURN p.name, p.address, p.property_type, p.location LIMIT 10",
                
                "User: What are the vacancy rates for retail properties?\nCypher: MATCH (p:Property)-[:HAS_VACANCY]->(v:Vacancy) WHERE p.property_type = 'Retail' RETURN p.name, p.address, v.vacancy_rate, v.last_updated ORDER BY v.vacancy_rate DESC",
                
                "User: Find properties with cap rates above 6%\nCypher: MATCH (p:Property)-[:HAS_FINANCIAL]->(f:Financial) WHERE f.cap_rate > 6.0 RETURN p.name, p.address, f.cap_rate, p.property_type ORDER BY f.cap_rate DESC",
                
                "User: List all properties managed by CBRE\nCypher: MATCH (p:Property)-[:MANAGED_BY]->(m:Management) WHERE m.manager = 'CBRE' RETURN p.name, p.address, m.manager, p.property_type",
                
                "User: Show me properties with recent lease renewals\nCypher: MATCH (p:Property)-[:HAS_LEASE]->(lease:Lease) WHERE lease.renewal_date >= date() - duration('P30D') RETURN p.name, p.address, lease.renewal_date, lease.tenant_name"
            ]
        
        return examples

    def _format_result(self, record: Record) -> RetrieverResultItem:
        """Format the result into RetrieverResultItem."""
        content = " | ".join(f"{k}: {v}" for k, v in record.items())
        metadata = record.data() if hasattr(record, "data") else {}
        return RetrieverResultItem(content=content, metadata=metadata)
