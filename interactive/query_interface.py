#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simple Query Interface - Clean and Maintainable
Follows DRY, SOLID, YAGNI principles
"""

import json
import re
import pyodbc
import time
import asyncio
from typing import List, Dict, Any, Optional
from datetime import datetime
from pathlib import Path

from langchain_openai import AzureChatOpenAI
from langchain.schema import HumanMessage, SystemMessage

from shared.config import Config
from shared.models import TableInfo, BusinessDomain, Relationship, QueryResult


class LLMClient:
    """Simple LLM client for SQL generation"""
    
    def __init__(self, config: Config):
        self.llm = AzureChatOpenAI(
            azure_endpoint=config.azure_endpoint,
            api_key=config.api_key,
            azure_deployment=config.deployment_name,
            api_version=config.api_version,
            # temperature=0.1,
            request_timeout=60
        )
    
    async def generate_sql(self, system_prompt: str, user_prompt: str) -> str:
        """Generate SQL from prompts"""
        try:
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt)
            ]
            response = await asyncio.to_thread(self.llm.invoke, messages)
            return response.content
        except Exception as e:
            print(f"   ⚠️ LLM error: {e}")
            return ""


class DataLoader:
    """Load and manage cached data"""
    
    def __init__(self, config: Config):
        self.config = config
        self.tables = []
        self.relationships = []
        self.domain = None
        self.database_structure = {}
    
    def load_data(self) -> bool:
        """Load all cached data"""
        try:
            # Load database structure
            db_file = self.config.get_cache_path("database_structure.json")
            if db_file.exists():
                with open(db_file, 'r', encoding='utf-8') as f:
                    self.database_structure = json.load(f)
            
            # Load semantic analysis
            semantic_file = self.config.get_cache_path("semantic_analysis.json")
            if semantic_file.exists():
                with open(semantic_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self._load_tables(data)
                    self._load_relationships(data)
                    self._load_domain(data)
            
            return len(self.tables) > 0
        except Exception as e:
            print(f"❌ Failed to load data: {e}")
            return False
    
    def _load_tables(self, data: Dict):
        """Load table information"""
        self.tables = []
        for table_data in data.get('tables', []):
            table = TableInfo(
                name=table_data['name'],
                schema=table_data['schema'],
                full_name=table_data['full_name'],
                object_type=table_data['object_type'],
                row_count=table_data['row_count'],
                columns=table_data['columns'],
                sample_data=table_data['sample_data'],
                relationships=table_data.get('relationships', [])
            )
            table.entity_type = table_data.get('entity_type', 'Unknown')
            table.confidence = table_data.get('confidence', 0.0)
            self.tables.append(table)
    
    def _load_relationships(self, data: Dict):
        """Load relationships"""
        self.relationships = []
        for rel_data in data.get('relationships', []):
            self.relationships.append(Relationship(
                from_table=rel_data['from_table'],
                to_table=rel_data['to_table'],
                relationship_type=rel_data['relationship_type'],
                confidence=rel_data['confidence'],
                description=rel_data.get('description', '')
            ))
    
    def _load_domain(self, data: Dict):
        """Load domain information"""
        domain_data = data.get('domain')
        if domain_data:
            self.domain = BusinessDomain(
                domain_type=domain_data['domain_type'],
                industry=domain_data['industry'],
                confidence=domain_data['confidence'],
                sample_questions=domain_data['sample_questions'],
                capabilities=domain_data['capabilities']
            )


class TableSelector:
    """Select relevant tables for queries with business intent analysis"""
    
    def __init__(self, tables: List[TableInfo], llm: LLMClient):
        self.tables = tables
        self.llm = llm
    
    async def find_relevant_tables(self, question: str) -> List[TableInfo]:
        """Find tables relevant to the question using business intent"""
        
        # First, analyze what the question is really asking for
        intent = self._analyze_business_intent(question)
        
        # Create enhanced table summaries with business context
        table_summaries = []
        for table in self.tables:
            sample_preview = self._create_sample_preview(table.sample_data)
            business_indicators = self._get_business_indicators(table, question)
            
            table_summaries.append({
                'full_name': table.full_name,
                'entity_type': table.entity_type,
                'confidence': table.confidence,
                'row_count': table.row_count,
                'columns': [col['name'] for col in table.columns[:10]],
                'sample_preview': sample_preview,
                'has_data': len(table.sample_data) > 0,
                'business_score': business_indicators['score'],
                'business_reason': business_indicators['reason']
            })
        
        # Sort by business relevance and confidence
        table_summaries.sort(key=lambda t: (t['business_score'], t['confidence'], t['row_count']), reverse=True)
        
        system_prompt = """
You are a business intelligence analyst. Select tables that contain the ACTUAL DATA needed to answer business questions.

For revenue questions: Look for tables with amount/price/revenue columns and transaction dates
For renewal questions: Look for tables with contract/subscription data and renewal/expiration dates  
For time-based questions: Look for tables with date columns matching the time period
For risk analysis: Look for tables with status, payment, or performance indicators

CRITICAL: Focus on transactional tables with actual business data, not just reference/lookup tables.
"""
        
        user_prompt = f"""
Question: "{question}"

Business Intent Analysis: {intent}

Available tables (sorted by business relevance):
{json.dumps(table_summaries[:20], indent=2)}

Select 4-8 tables that contain the ACTUAL DATA needed to answer this specific business question.
Prioritize tables with:
1. Relevant date columns for time periods
2. Amount/revenue columns for financial questions  
3. Status/renewal columns for risk analysis
4. Transaction data over reference data

JSON format:
{{
  "selected_tables": ["[schema].[table1]", "[schema].[table2]"],
  "reasoning": "Specific explanation of how these tables will answer the business question"
}}
"""
        
        response = await self.llm.generate_sql("", user_prompt)
        result = self._parse_json(response)
        
        if result and 'selected_tables' in result:
            selected_names = result['selected_tables']
            selected_tables = [t for t in self.tables if t.full_name in selected_names]
            print(f"      🎯 Selected based on business intent: {intent}")
            return selected_tables
        
        # Enhanced fallback: prioritize business-relevant tables
        return self._smart_fallback_selection(question)
    
    def _analyze_business_intent(self, question: str) -> str:
        """Analyze what the business question is asking for"""
        q = question.lower()
        
        intents = []
        
        # Revenue/Financial analysis
        if any(word in q for word in ['revenue', 'income', 'sales', 'profit', 'amount', 'value', 'financial']):
            intents.append("financial_analysis")
        
        # Time-based analysis
        if any(word in q for word in ['quarter', 'month', 'year', 'next', 'last', 'this', 'period']):
            intents.append("time_based_analysis")
        
        # Risk/Renewal analysis
        if any(word in q for word in ['risk', 'renewal', 'expire', 'cancel', 'churn', 'retention']):
            intents.append("risk_analysis")
        
        # Customer analysis
        if any(word in q for word in ['customer', 'client', 'account', 'subscriber']):
            intents.append("customer_analysis")
        
        # Contract/Subscription analysis
        if any(word in q for word in ['contract', 'subscription', 'agreement', 'deal']):
            intents.append("contract_analysis")
        
        return " + ".join(intents) if intents else "general_query"
    
    def _get_business_indicators(self, table: TableInfo, question: str) -> Dict[str, Any]:
        """Score table relevance for business question"""
        score = 0
        reasons = []
        
        table_name = table.full_name.lower()
        column_names = [col['name'].lower() for col in table.columns]
        q = question.lower()
        
        # Revenue/Financial indicators
        if any(word in q for word in ['revenue', 'amount', 'financial', 'money']):
            if any(col in column_names for col in ['amount', 'price', 'total', 'revenue', 'value']):
                score += 3
                reasons.append("has_financial_columns")
            if any(word in table_name for word in ['payment', 'invoice', 'billing', 'financial']):
                score += 2
                reasons.append("financial_table")
        
        # Renewal/Contract indicators  
        if any(word in q for word in ['renewal', 'contract', 'subscription', 'expire']):
            if any(col in column_names for col in ['renewaldate', 'expirationdate', 'contractdate', 'subscriptiondate']):
                score += 3
                reasons.append("has_renewal_dates")
            if any(word in table_name for word in ['contract', 'subscription', 'renewal']):
                score += 2
                reasons.append("contract_table")
        
        # Time-based indicators
        if any(word in q for word in ['quarter', 'next', 'period', 'time']):
            date_columns = [col for col in column_names if 'date' in col or 'time' in col]
            if date_columns:
                score += 2
                reasons.append(f"has_date_columns: {date_columns[:3]}")
        
        # Risk indicators
        if any(word in q for word in ['risk', 'why', 'reason']):
            if any(col in column_names for col in ['status', 'active', 'cancelled', 'reason']):
                score += 2
                reasons.append("has_status_columns")
        
        # Data quality bonus
        if table.sample_data and len(table.sample_data) > 0:
            score += 1
            reasons.append("has_sample_data")
        
        return {
            'score': score,
            'reason': ', '.join(reasons) if reasons else 'no_specific_indicators'
        }
    
    def _smart_fallback_selection(self, question: str) -> List[TableInfo]:
        """Smart fallback when LLM selection fails"""
        intent = self._analyze_business_intent(question)
        
        # Score all tables by business relevance
        scored_tables = []
        for table in self.tables:
            indicators = self._get_business_indicators(table, question)
            scored_tables.append((table, indicators['score']))
        
        # Sort by business score then confidence
        scored_tables.sort(key=lambda x: (x[1], x[0].confidence), reverse=True)
        
        # Return top scored tables
        return [table for table, score in scored_tables[:6] if score > 0]
    
    def _create_sample_preview(self, sample_data: List[Dict]) -> str:
        """Create preview of sample data"""
        if not sample_data:
            return "No sample data"
        
        first_row = sample_data[0]
        preview_items = []
        
        for key, value in list(first_row.items())[:4]:
            if value is not None:
                value_str = str(value)[:20]
                preview_items.append(f"{key}={value_str}")
        
        return ", ".join(preview_items) if preview_items else "Empty data"
    
    def _parse_json(self, response: str) -> Dict:
        """Parse JSON from response"""
        try:
            cleaned = re.sub(r'```json\s*', '', response)
            cleaned = re.sub(r'```\s*', '', cleaned)
            cleaned = re.sub(r'^[^{]*', '', cleaned)
            cleaned = re.sub(r'[^}]*$', '', cleaned)
            return json.loads(cleaned)
        except:
            return {}


class RelationshipMapper:
    """Map relationships between tables"""
    
    def __init__(self, database_structure: Dict):
        self.database_structure = database_structure
        self.foreign_keys = self._build_foreign_key_map()
    
    def _build_foreign_key_map(self) -> Dict[str, List[Dict]]:
        """Build foreign key mapping"""
        fk_map = {}
        
        # From explicit relationships
        for rel in self.database_structure.get('relationships', []):
            from_table = rel.get('from_table', '')
            if from_table:
                fk_map.setdefault(from_table, []).append(rel)
        
        # From table-level relationships
        for table_data in self.database_structure.get('tables', []):
            table_name = table_data.get('full_name', '')
            for fk_info in table_data.get('relationships', []):
                if '->' in fk_info:
                    parts = fk_info.split(' -> ')
                    if len(parts) == 2:
                        from_col = parts[0].strip()
                        to_info = parts[1].strip()
                        
                        fk_map.setdefault(table_name, []).append({
                            'from_table': table_name,
                            'from_column': from_col,
                            'to_table': to_info.split('.')[0] if '.' in to_info else to_info,
                            'to_column': to_info.split('.')[1] if '.' in to_info else 'ID'
                        })
        
        return fk_map
    
    def find_relationships(self, tables: List[TableInfo]) -> List[Dict]:
        """Find relationships between selected tables"""
        table_names = [t.full_name for t in tables]
        relationships = []
        
        for table_name in table_names:
            for fk in self.foreign_keys.get(table_name, []):
                target_table = fk.get('to_table', '')
                
                # Check if target table is in our selected tables
                if any(target_table in t.full_name for t in tables):
                    relationships.append({
                        'from_table': table_name,
                        'to_table': target_table,
                        'from_column': fk.get('from_column', ''),
                        'to_column': fk.get('to_column', ''),
                        'join_type': 'INNER JOIN'
                    })
        
        return relationships


class SQLGenerator:
    """Generate SQL queries with business intelligence"""
    
    def __init__(self, llm: LLMClient):
        self.llm = llm
    
    async def generate_query(self, question: str, tables: List[TableInfo], 
                           relationships: List[Dict]) -> str:
        """Generate SQL query with business context"""
        
        # Analyze business intent for SQL generation
        business_context = self._analyze_sql_context(question, tables)
        
        # Prepare enhanced table context
        table_context = []
        for table in tables:
            columns_info = []
            date_columns = []
            amount_columns = []
            status_columns = []
            
            for col in table.columns[:15]:  # More columns for better context
                col_name = col['name']
                col_type = col['data_type']
                columns_info.append(f"{col_name} ({col_type})")
                
                # Categorize important columns
                if any(word in col_name.lower() for word in ['date', 'time']):
                    date_columns.append(col_name)
                if any(word in col_name.lower() for word in ['amount', 'price', 'total', 'revenue', 'value']):
                    amount_columns.append(col_name)
                if any(word in col_name.lower() for word in ['status', 'active', 'cancelled', 'renewed']):
                    status_columns.append(col_name)
            
            # Enhanced sample data analysis
            sample_analysis = self._analyze_sample_data(table.sample_data)
            
            table_context.append({
                'table_name': table.full_name,
                'entity_type': table.entity_type,
                'columns': columns_info,
                'date_columns': date_columns,
                'amount_columns': amount_columns,
                'status_columns': status_columns,
                'sample_analysis': sample_analysis,
                'row_count': table.row_count
            })
        
        system_prompt = f"""
You are a business intelligence SQL expert specializing in {business_context['domain']}.

BUSINESS CONTEXT: {business_context['intent']}

CRITICAL RULES:
- For REVENUE questions: Always include SUM() of amount/price columns
- For TIME-PERIOD questions: Use proper date filtering (DATEADD, GETDATE())
- For RENEWAL questions: Look for renewal/expiration dates and contract values
- For RISK questions: Include reasons (status, payment issues, etc.)
- Use proper JOINs with the verified relationships provided
- Return TOP 100 to limit results
- Use clear column aliases for business meaning
- No variables (@var), use inline expressions only

BUSINESS QUERY PATTERNS:
- Revenue at risk: SUM(contract_value) WHERE renewal_date BETWEEN ... AND ... AND risk_factors
- Next quarter: DATEADD(quarter, 1, GETDATE()) and DATEADD(quarter, 2, GETDATE())
- Customer analysis: GROUP BY customer with aggregations
- Financial reporting: SUM, COUNT, percentages with business-friendly names

Return ONLY the SQL query that directly answers the business question.
"""
        
        user_prompt = f"""
BUSINESS QUESTION: "{question}"

AVAILABLE TABLES WITH BUSINESS CONTEXT:
{json.dumps(table_context, indent=2)}

VERIFIED RELATIONSHIPS FOR JOINS:
{json.dumps(relationships, indent=2)}

BUSINESS REQUIREMENTS:
{business_context['requirements']}

Generate a SQL query that directly answers this business question.
Focus on the specific metrics, time periods, and business logic required.
"""
        
        response = await self.llm.generate_sql(system_prompt, user_prompt)
        return self._clean_sql(response)
    
    def _analyze_sql_context(self, question: str, tables: List[TableInfo]) -> Dict[str, str]:
        """Analyze business context for SQL generation"""
        q = question.lower()
        
        # Determine domain
        entity_types = [t.entity_type for t in tables]
        if 'Financial' in entity_types or 'Payment' in entity_types:
            domain = "Financial/Revenue Analysis"
        elif 'Customer' in entity_types:
            domain = "Customer Relationship Management"
        else:
            domain = "Business Operations"
        
        # Determine intent and requirements
        if 'revenue' in q and 'risk' in q:
            intent = "Revenue Risk Analysis"
            requirements = [
                "Calculate total revenue amounts at risk",
                "Identify time period (next quarter)",
                "Include risk reasons/explanations",
                "Group by relevant business dimensions"
            ]
        elif 'renewal' in q:
            intent = "Renewal Analysis"
            requirements = [
                "Focus on renewal/expiration dates",
                "Include contract/subscription values",
                "Calculate time-based filtering",
                "Identify renewal status and risks"
            ]
        elif any(word in q for word in ['revenue', 'sales', 'financial']):
            intent = "Financial Analysis"
            requirements = [
                "Sum/aggregate financial amounts",
                "Apply date filtering for time periods",
                "Include business-friendly groupings"
            ]
        else:
            intent = "General Business Query"
            requirements = [
                "Focus on business metrics",
                "Include relevant filters and groupings"
            ]
        
        return {
            'domain': domain,
            'intent': intent,
            'requirements': '; '.join(requirements)
        }
    
    def _analyze_sample_data(self, sample_data: List[Dict]) -> str:
        """Analyze sample data for SQL context"""
        if not sample_data:
            return "No sample data available"
        
        analysis = []
        first_row = sample_data[0]
        
        # Look for key business indicators
        for key, value in first_row.items():
            if value is not None:
                key_lower = key.lower()
                
                # Financial indicators
                if any(word in key_lower for word in ['amount', 'price', 'total', 'revenue']):
                    analysis.append(f"Financial: {key}={value}")
                
                # Date indicators  
                elif any(word in key_lower for word in ['date', 'time']):
                    analysis.append(f"Date: {key}={value}")
                
                # Status indicators
                elif any(word in key_lower for word in ['status', 'active', 'cancelled']):
                    analysis.append(f"Status: {key}={value}")
        
        return "; ".join(analysis[:5]) if analysis else "General data fields"
    
    def _clean_sql(self, response: str) -> str:
        """Clean SQL from response"""
        # Remove markdown
        cleaned = re.sub(r'```sql\s*', '', response, flags=re.IGNORECASE)
        cleaned = re.sub(r'```\s*', '', cleaned)
        
        # Extract SQL
        lines = cleaned.strip().split('\n')
        sql_lines = []
        
        for line in lines:
            line = line.strip()
            if line.upper().startswith(('SELECT', 'WITH')) or sql_lines:
                sql_lines.append(line)
        
        return '\n'.join(sql_lines).rstrip(';').strip()


class QueryExecutor:
    """Execute SQL queries safely"""
    
    def __init__(self, config: Config):
        self.config = config
    
    def execute_query(self, sql: str) -> tuple:
        """Execute SQL query and return results"""
        if not sql or not sql.strip():
            return [], "Empty SQL query"
        
        try:
            with pyodbc.connect(self.config.get_database_connection_string()) as conn:
                # Set UTF-8 encoding for Greek text
                conn.setdecoding(pyodbc.SQL_CHAR, encoding='utf-8')
                conn.setdecoding(pyodbc.SQL_WCHAR, encoding='utf-8')
                conn.setencoding(encoding='utf-8')
                
                cursor = conn.cursor()
                cursor.execute(sql)
                
                if cursor.description:
                    columns = [col[0] for col in cursor.description]
                    results = []
                    
                    for row in cursor.fetchmany(100):  # Limit to 100 rows
                        row_dict = {}
                        for i, value in enumerate(row):
                            if i < len(columns):
                                row_dict[columns[i]] = self._safe_value(value)
                        results.append(row_dict)
                    
                    return results, None
                else:
                    return [], None
                    
        except Exception as e:
            error_msg = str(e)
            if "Invalid column name" in error_msg:
                return [], f"Column not found: {error_msg}"
            elif "Invalid object name" in error_msg:
                return [], f"Table not found: {error_msg}"
            else:
                return [], f"SQL Error: {error_msg}"
    
    def _safe_value(self, value):
        """Convert value to safe format"""
        if value is None:
            return None
        elif isinstance(value, datetime):
            return value.isoformat()
        elif isinstance(value, (str, int, float, bool)):
            return value
        else:
            return str(value)[:200]


class QueryInterface:
    """Main query interface orchestrator"""
    
    def __init__(self, config: Config):
        self.config = config
        self.llm = LLMClient(config)
        self.data_loader = DataLoader(config)
        self.executor = QueryExecutor(config)
        
        # Load cached data
        if not self.data_loader.load_data():
            raise ValueError("Failed to load cached data. Run discovery and analysis first.")
        
        self.table_selector = TableSelector(self.data_loader.tables, self.llm)
        self.relationship_mapper = RelationshipMapper(self.data_loader.database_structure)
        self.sql_generator = SQLGenerator(self.llm)
        
        print(f"   📊 Loaded {len(self.data_loader.tables)} tables")
        print(f"   🔗 Database relationships: {len(self.relationship_mapper.foreign_keys)}")
    
    async def start_interactive_session(self, tables: List[TableInfo], 
                                      domain: Optional[BusinessDomain], 
                                      relationships: List[Relationship]):
        """Start interactive query session"""
        
        print(f"🚀 Enhanced 4-Stage Pipeline Ready")
        print(f"   📊 Classified tables: {len(tables)}")
        
        if domain:
            print(f"   🏢 Domain: {domain.domain_type}")
        
        # Show entity distribution
        entity_counts = {}
        for table in tables:
            if table.entity_type != 'Unknown':
                entity_counts[table.entity_type] = entity_counts.get(table.entity_type, 0) + 1
        
        if entity_counts:
            print(f"   📊 Available entities: {dict(list(entity_counts.items())[:5])}")
        
        query_count = 0
        
        while True:
            try:
                question = input(f"\n❓ Query #{query_count + 1}: ").strip()
                
                if question.lower() in ['quit', 'exit', 'q']:
                    break
                if not question:
                    continue
                
                query_count += 1
                print(f"🚀 Processing with enhanced 4-stage pipeline...")
                
                start_time = time.time()
                result = await self.process_query(question)
                result.execution_time = time.time() - start_time
                
                self.display_result(result)
                
            except KeyboardInterrupt:
                print("\n⏸️ Interrupted")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
        
        print(f"\n📊 Session summary: {query_count} queries processed")
    
    async def process_query(self, question: str) -> QueryResult:
        """Process a single query through the 4-stage pipeline with intelligent retry"""
        
        try:
            # Stage 1: Understand intent (implicit in question analysis)
            print("   🎯 Stage 1: Understanding intent...")
            
            # Stage 2: Find relevant tables
            print("   📋 Stage 2: Finding tables using real sample data...")
            selected_tables = await self.table_selector.find_relevant_tables(question)
            
            if not selected_tables:
                return QueryResult(
                    question=question,
                    sql_query="",
                    results=[],
                    error="No relevant tables found"
                )
            
            print(f"      ✅ Found {len(selected_tables)} relevant tables")
            
            # Stage 3: Discover relationships
            print("   🔗 Stage 3: Analyzing actual database relationships...")
            relationships = self.relationship_mapper.find_relationships(selected_tables)
            print(f"      🔗 Found {len(relationships)} verified relationships")
            
            # Stage 4: Generate and execute SQL (with retry logic)
            print("   ⚡ Stage 4: Generating and executing SQL...")
            
            # First attempt
            sql = await self.sql_generator.generate_query(question, selected_tables, relationships)
            
            if not sql:
                return QueryResult(
                    question=question,
                    sql_query="",
                    results=[],
                    error="Failed to generate SQL query"
                )
            
            # Validate SQL answers the business question
            validation_result = self._validate_business_logic(question, sql, selected_tables)
            if validation_result['issues']:
                print(f"      ⚠️ Business validation issues: {', '.join(validation_result['issues'])}")
                print("      🔄 Regenerating with business context...")
                
                # Regenerate with validation feedback
                sql = await self._regenerate_with_context(question, selected_tables, relationships, validation_result['issues'])
            
            # Execute query
            results, error = self.executor.execute_query(sql)
            
            # **NEW: Intelligent Result Validation & Auto-Retry**
            if not error:
                result_analysis = await self._analyze_result_quality(question, sql, results, selected_tables)
                
                if result_analysis['needs_retry']:
                    print(f"      🔍 Result analysis: {result_analysis['issue']}")
                    print("      🔄 Auto-retrying with improved logic...")
                    
                    # Retry with result-based improvements
                    improved_sql = await self._retry_with_result_analysis(
                        question, selected_tables, relationships, sql, results, result_analysis
                    )
                    
                    if improved_sql and improved_sql != sql:
                        print("      ✅ Generated improved query")
                        retry_results, retry_error = self.executor.execute_query(improved_sql)
                        
                        # Use retry results if they're better
                        if not retry_error and len(retry_results) > 0:
                            print(f"      🎯 Retry successful: {len(retry_results)} rows vs {len(results)} rows")
                            sql = improved_sql
                            results = retry_results
                            error = retry_error
                        else:
                            print(f"      ⚠️ Retry didn't improve results, using original")
            
            return QueryResult(
                question=question,
                sql_query=sql,
                results=results,
                error=error,
                tables_used=[t.full_name for t in selected_tables]
            )
            
        except Exception as e:
            return QueryResult(
                question=question,
                sql_query="",
                results=[],
                error=f"Pipeline failed: {str(e)}"
            )
    
    async def _analyze_result_quality(self, question: str, sql: str, results: List[Dict], 
                                    tables: List[TableInfo]) -> Dict[str, Any]:
        """Analyze if results make business sense and suggest improvements"""
        
        q = question.lower()
        issues = []
        needs_retry = False
        
        # Check for empty results on business questions
        if len(results) == 0:
            if any(word in q for word in ['revenue', 'customer', 'payment', 'order', 'contract']):
                issues.append("Empty results for business question - likely date filter or join issue")
                needs_retry = True
        
        # Check for unrealistic single values
        elif len(results) == 1 and len(results[0]) == 1:
            value = list(results[0].values())[0]
            
            # Revenue should typically be > 0
            if 'revenue' in q and (value is None or value == 0):
                issues.append("Zero revenue result - check amount columns and filters")
                needs_retry = True
            
            # Customer counts should be reasonable
            if 'customer' in q and 'count' in q and (value is None or value == 0):
                issues.append("Zero customer count - check customer tables and criteria")
                needs_retry = True
        
        # Check for missing time period filtering
        if any(period in q for period in ['quarter', 'month', 'year', 'next', 'last']):
            sql_lower = sql.lower()
            if not any(func in sql_lower for func in ['dateadd', 'datediff', 'getdate', 'year(', 'month(']):
                issues.append("Time period mentioned but no date filtering detected")
                needs_retry = True
        
        # Check for potential join issues (lots of relationships but simple query)
        if len(results) == 0 and 'join' in sql.lower():
            issues.append("Complex joins resulted in empty set - may need LEFT JOINs or simpler approach")
            needs_retry = True
        
        return {
            'needs_retry': needs_retry,
            'issue': '; '.join(issues) if issues else 'Results look reasonable',
            'result_count': len(results),
            'has_data': len(results) > 0
        }
    
    async def _retry_with_result_analysis(self, question: str, tables: List[TableInfo], 
                                        relationships: List[Dict], original_sql: str, 
                                        original_results: List[Dict], analysis: Dict) -> str:
        """Generate improved SQL based on result analysis"""
        
        # Analyze what went wrong and prepare corrections
        corrections = []
        q = question.lower()
        
        # If empty results, suggest fixes
        if len(original_results) == 0:
            corrections.append("Use LEFT JOINs instead of INNER JOINs to preserve base records")
            corrections.append("Check date range filters - make them wider if too restrictive")
            corrections.append("Verify table relationships and consider single-table approach if joins fail")
            
            # For time-based queries, suggest alternative date logic
            if 'next quarter' in q:
                corrections.append("Try alternative next quarter logic: WHERE DATEPART(quarter, date_col) = DATEPART(quarter, DATEADD(quarter, 1, GETDATE()))")
        
        # Prepare sample data insights for better column selection
        table_insights = []
        for table in tables:
            if table.sample_data:
                sample = table.sample_data[0]
                date_fields = []
                amount_fields = []
                
                for col_name, value in sample.items():
                    if value is not None:
                        col_lower = col_name.lower()
                        if any(word in col_lower for word in ['date', 'time']) and 'date' in str(type(value)).lower():
                            date_fields.append(col_name)
                        if any(word in col_lower for word in ['amount', 'price', 'total', 'value']) and isinstance(value, (int, float)):
                            amount_fields.append(f"{col_name}={value}")
                
                table_insights.append({
                    'table': table.full_name,
                    'date_fields': date_fields,
                    'amount_fields': amount_fields,
                    'sample_count': len(table.sample_data)
                })
        
        system_prompt = f"""
You are a SQL troubleshooting expert. The previous query returned {len(original_results)} rows, which doesn't make business sense.

ANALYSIS: {analysis['issue']}

CORRECTIONS TO APPLY:
{chr(10).join([f"- {correction}" for correction in corrections])}

KEY PRINCIPLES:
1. If INNER JOINs return 0 rows, try LEFT JOINs or single table approach
2. If date filtering is too restrictive, widen the range or use different date columns
3. Focus on tables with actual sample data showing realistic values
4. For revenue questions, ensure you're using columns with actual monetary values

SAMPLE DATA INSIGHTS:
{json.dumps(table_insights, indent=2)}

Generate IMPROVED SQL that addresses the analysis issues and returns meaningful business results.
Return ONLY the SQL query.
"""
        
        user_prompt = f"""
BUSINESS QUESTION: "{question}"

ORIGINAL SQL THAT FAILED:
{original_sql}

AVAILABLE TABLES:
{json.dumps([{'table': t.full_name, 'entity': t.entity_type, 'rows': t.row_count} for t in tables], indent=2)}

RELATIONSHIPS:
{json.dumps(relationships, indent=2)}

Generate improved SQL that fixes the issues and returns meaningful business data.
"""
        
        response = await self.llm.generate_sql(system_prompt, user_prompt)
        return self.sql_generator._clean_sql(response)
    
    def _validate_business_logic(self, question: str, sql: str, tables: List[TableInfo]) -> Dict[str, Any]:
        """Validate that SQL addresses the business question"""
        q = question.lower()
        sql_lower = sql.lower()
        issues = []
        
        # Revenue questions should have aggregations
        if 'revenue' in q and not any(func in sql_lower for func in ['sum(', 'sum ', 'total']):
            issues.append("Missing revenue calculation (SUM)")
        
        # Time period questions should have date filtering
        if any(period in q for period in ['quarter', 'month', 'year', 'next', 'last']) and not any(func in sql_lower for func in ['dateadd', 'getdate', 'where']):
            issues.append("Missing time period filtering")
        
        # Risk questions should explain why
        if 'why' in q or 'reason' in q:
            if not any(word in sql_lower for word in ['case', 'reason', 'status', 'why']):
                issues.append("Missing risk explanation (CASE statement or reason columns)")
        
        # Renewal questions should focus on renewal/expiration dates
        if 'renewal' in q or 'expire' in q:
            table_columns = []
            for table in tables:
                table_columns.extend([col['name'].lower() for col in table.columns])
            
            renewal_columns = [col for col in table_columns if 'renewal' in col or 'expir' in col or 'contract' in col]
            if renewal_columns and not any(col in sql_lower for col in renewal_columns):
                issues.append(f"Missing renewal date columns: {renewal_columns[:3]}")
        
        return {
            'issues': issues,
            'has_issues': len(issues) > 0
        }
    
    async def _regenerate_with_context(self, question: str, tables: List[TableInfo], 
                                     relationships: List[Dict], issues: List[str]) -> str:
        """Regenerate SQL with validation feedback"""
        
        # Prepare business-focused context
        business_requirements = []
        q = question.lower()
        
        if 'revenue' in q:
            amount_columns = []
            for table in tables:
                for col in table.columns:
                    if any(word in col['name'].lower() for word in ['amount', 'price', 'total', 'revenue']):
                        amount_columns.append(f"{table.full_name}.{col['name']}")
            business_requirements.append(f"MUST include SUM() of amount columns: {amount_columns[:3]}")
        
        if 'next quarter' in q:
            business_requirements.append("MUST include next quarter date filtering: WHERE date_column >= DATEADD(quarter, 1, GETDATE()) AND date_column < DATEADD(quarter, 2, GETDATE())")
        
        if 'why' in q or 'reason' in q:
            business_requirements.append("MUST include CASE statement explaining reasons/risk factors")
        
        system_prompt = f"""
You are a business intelligence expert. The previous SQL had these issues: {', '.join(issues)}

CRITICAL BUSINESS REQUIREMENTS:
{chr(10).join(business_requirements)}

Generate SQL that directly answers the business question with proper:
- Financial calculations (SUM, aggregations)
- Date filtering for time periods  
- Business explanations (CASE statements)
- Relevant grouping and ordering

Return ONLY valid T-SQL that addresses ALL business requirements.
"""
        
        table_context = []
        for table in tables:
            # Focus on business-relevant columns
            important_columns = []
            for col in table.columns:
                col_name = col['name']
                if any(word in col_name.lower() for word in ['amount', 'price', 'total', 'revenue', 'date', 'time', 'status', 'renewal', 'expir']):
                    important_columns.append(f"{col_name} ({col['data_type']})")
            
            table_context.append({
                'table_name': table.full_name,
                'entity_type': table.entity_type,
                'important_columns': important_columns,
                'row_count': table.row_count
            })
        
        user_prompt = f"""
BUSINESS QUESTION: "{question}"

VALIDATION ISSUES TO FIX: {', '.join(issues)}

BUSINESS-FOCUSED TABLES:
{json.dumps(table_context, indent=2)}

VERIFIED RELATIONSHIPS:
{json.dumps(relationships, indent=2)}

Generate SQL that DIRECTLY ANSWERS the business question and fixes all validation issues.
"""
        
        response = await self.llm.generate_sql(system_prompt, user_prompt)
        return self.sql_generator._clean_sql(response)
    
    def display_result(self, result: QueryResult):
        """Display query result with retry information"""
        
        print(f"⏱️ Completed in {result.execution_time:.1f}s")
        print("-" * 60)
        
        if result.error:
            print(f"❌ Error: {result.error}")
            if result.sql_query:
                print(f"📋 Generated SQL:\n   {result.sql_query}")
        else:
            print(f"📋 Generated SQL:\n   {result.sql_query}")
            print(f"📊 Results: {len(result.results)} rows")
            
            if result.results:
                # Handle single value results
                if result.is_single_value():
                    value = result.get_single_value()
                    column_name = list(result.results[0].keys())[0]
                    
                    if isinstance(value, (int, float)):
                        print(f"   🎯 {column_name}: {value:,}")
                    else:
                        print(f"   🎯 {column_name}: {value}")
                else:
                    # Handle multiple rows
                    for i, row in enumerate(result.results[:5], 1):
                        display_row = {}
                        for key, value in list(row.items())[:6]:
                            if isinstance(value, str) and len(value) > 30:
                                display_row[key] = value[:30] + "..."
                            elif isinstance(value, (int, float)) and value > 1000:
                                display_row[key] = f"{value:,}"
                            else:
                                display_row[key] = value
                        print(f"   {i}. {display_row}")
                    
                    if len(result.results) > 5:
                        print(f"   ... and {len(result.results) - 5} more rows")
            else:
                # Enhanced empty result analysis
                print("   ⚠️ No results returned")
                print("   💡 This might indicate:")
                print("      • Date filters are too restrictive")
                print("      • Table joins are not matching data")
                print("      • Selected tables don't contain relevant data")
                
                # Check if retry was attempted (we can detect this from execution time or other indicators)
                if result.execution_time > 40:  # Likely included a retry
                    print("   🔄 Auto-retry was attempted to improve results")
            
            # Show tables used with enhanced context
            if result.tables_used:
                print(f"📋 Tables used:")
                for table_name in result.tables_used:
                    # Find entity type and row count
                    entity_type = "Unknown"
                    row_count = 0
                    for table in self.data_loader.tables:
                        if table.full_name == table_name:
                            entity_type = table.entity_type
                            row_count = table.row_count
                            break
                    print(f"      • {table_name} ({entity_type}) - {row_count:,} rows")
            
            # Show relationship status
            has_joins = 'JOIN' in (result.sql_query or '').upper()
            if has_joins:
                print("✅ Query used verified database relationships")
            else:
                print("ℹ️ Single-table query (no joins needed)")
        
        # Enhanced debugging help
        print("\n💡 Debugging Help:")
        print("   • Check database_structure.json for table/column information")
        print("   • Check semantic_analysis.json for entity classifications")
        print("   • Foreign key relationships are automatically discovered and used")
        
        if not result.error and len(result.results) == 0:
            print("   🔍 For empty results, try:")
            print("      • Broader date ranges (e.g., 'this year' instead of 'next quarter')")
            print("      • Different table combinations")
            print("      • Simpler questions to verify data exists")
            
        # Business question suggestions
        if hasattr(result, 'tables_used') and result.tables_used:
            print("   💼 Try related business questions:")
            entity_types = set()
            for table_name in result.tables_used:
                for table in self.data_loader.tables:
                    if table.full_name == table_name:
                        entity_types.add(table.entity_type)
                        break
            
            suggestions = []
            if 'Customer' in entity_types:
                suggestions.append("'How many customers do we have?'")
            if 'Payment' in entity_types or 'Financial' in entity_types:
                suggestions.append("'What is our total revenue this year?'")
            if 'Order' in entity_types:
                suggestions.append("'How many orders were placed last month?'")
            
            if suggestions:
                print(f"      {', '.join(suggestions[:3])}")