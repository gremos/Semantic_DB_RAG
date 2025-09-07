#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Query Interface - LLM-Driven Intelligence
Simple, Readable, Maintainable - DRY, SOLID, YAGNI principles
More LLM intelligence, less rigid coding
"""

import asyncio
import pyodbc
import time
from typing import List, Dict, Any, Optional

from langchain_openai import AzureChatOpenAI
from langchain.schema import HumanMessage, SystemMessage

# SQLGlot for safety validation (Architecture requirement)
try:
    import sqlglot
    HAS_SQLGLOT = True
except ImportError:
    HAS_SQLGLOT = False

from shared.config import Config
from shared.models import TableInfo, BusinessDomain, Relationship, QueryResult
from shared.utils import parse_json_response, clean_sql_query, safe_database_value

class IntelligentAnalyzer:
    """LLM-driven intelligent query analysis - no rigid entity constraints"""
    
    def __init__(self, llm: AzureChatOpenAI):
        self.llm = llm
    
    async def analyze_and_generate_sql(self, question: str, tables: List[TableInfo]) -> Dict[str, Any]:
        """Single LLM call for complete analysis and SQL generation"""
        print("   🧠 LLM-driven intelligent analysis...")
        
        # Build comprehensive context
        context = self._build_intelligent_context(tables)
        
        # Single intelligent prompt for everything
        result = await self._get_intelligent_response(question, context)
        
        if result:
            print(f"      ✅ Intent: {result.get('intent_summary', 'analyzed')}")
            selected_tables = result.get('selected_tables', [])
            if selected_tables:
                print(f"      📊 Selected: {', '.join([t.get('table_name', 'Unknown') for t in selected_tables])}")
            return result
        
        return self._create_fallback_response(question, tables)
    
    def _build_intelligent_context(self, tables: List[TableInfo]) -> str:
        """Build rich context for LLM understanding"""
        context_lines = [
            "DATABASE CONTEXT:",
            f"Total available tables: {len(tables)}",
            ""
        ]
        
        # Group tables by likely business area (simple heuristics)
        business_areas = {
            'Customer Data': [],
            'Financial Data': [], 
            'Sales Data': [],
            'Employee Data': [],
            'System Data': []
        }
        
        for table in tables:
            name_lower = table.name.lower()
            
            # Simple business area classification
            if any(word in name_lower for word in ['customer', 'client', 'account', 'contact']):
                business_areas['Customer Data'].append(table)
            elif any(word in name_lower for word in ['payment', 'transaction', 'invoice', 'billing', 'revenue']):
                business_areas['Financial Data'].append(table)
            elif any(word in name_lower for word in ['sales', 'deal', 'opportunity', 'contract', 'order']):
                business_areas['Sales Data'].append(table)
            elif any(word in name_lower for word in ['user', 'employee', 'staff', 'rep']):
                business_areas['Employee Data'].append(table)
            else:
                business_areas['System Data'].append(table)
        
        # Present tables by business area with rich details
        for area, area_tables in business_areas.items():
            if area_tables:
                context_lines.append(f"\n{area.upper()}:")
                for table in area_tables[:5]:  # Limit to keep context manageable
                    context_lines.append(f"\n  📊 {table.full_name}")
                    context_lines.append(f"      Rows: {table.row_count:,}")
                    
                    # Show meaningful columns
                    col_details = []
                    for col in table.columns[:8]:
                        col_name = col.get('name', '')
                        col_type = col.get('data_type', '')
                        
                        # Add business meaning hints
                        hints = []
                        if any(word in col_name.lower() for word in ['name', 'title', 'description']):
                            hints.append('DISPLAY')
                        if any(word in col_name.lower() for word in ['amount', 'total', 'revenue', 'price']):
                            hints.append('MONEY')
                        if any(word in col_name.lower() for word in ['date', 'time', 'created', 'modified']):
                            hints.append('DATE')
                        if col_name.lower().endswith('id') or 'id' in col_name.lower():
                            hints.append('ID')
                        
                        hint_str = f" ({', '.join(hints)})" if hints else ""
                        col_details.append(f"{col_name} ({col_type}){hint_str}")
                    
                    context_lines.append(f"      Columns: {', '.join(col_details)}")
                    
                    # Show sample data insights
                    if table.sample_data:
                        sample_insights = []
                        first_row = table.sample_data[0]
                        for key, value in list(first_row.items())[:4]:
                            if not key.startswith('__') and value is not None:
                                if isinstance(value, str) and len(str(value)) > 50:
                                    value = str(value)[:50] + "..."
                                sample_insights.append(f"{key}={value}")
                        
                        if sample_insights:
                            context_lines.append(f"      Sample: {', '.join(sample_insights)}")
        
        return '\n'.join(context_lines)
    
    async def _get_intelligent_response(self, question: str, context: str) -> Optional[Dict[str, Any]]:
        """Get intelligent response from LLM"""
        try:
            prompt = f"""You are a business intelligence expert analyzing a database to answer this question:

QUESTION: "{question}"

{context}

Your task is to:
1. Understand what the user is asking for
2. Identify which tables contain the data needed
3. Determine which columns to use for the query
4. Generate appropriate SQL Server T-SQL

Think step by step:
- What business entities does this question involve? (customers, sales reps, revenue, contracts, etc.)
- Which tables are most likely to contain this data?
- What columns would contain the specific data needed?
- What type of query is this? (ranking, aggregation, list, count)

IMPORTANT SQL GENERATION RULES:
- Generate SIMPLE SQL that works with a single table when possible
- Use JOINs only when absolutely necessary
- Prefer tables that already contain the needed data aggregated
- Use proper SQL Server syntax with square brackets [column_name]
- For ranking queries: SELECT TOP (N) ... ORDER BY ... DESC
- For aggregation: SELECT SUM/COUNT/AVG(...) FROM single_table WHERE ...
- Keep queries simple and safe

Respond with JSON only:
{{
  "intent_summary": "Brief description of what user wants",
  "business_entities": ["customer", "revenue", "sales_rep"],
  "query_type": "ranking|aggregation|list|count",
  "selected_tables": [
    {{
      "table_name": "[schema].[table]",
      "reason": "why this table was selected",
      "columns_needed": [
        {{
          "column_name": "column_name",
          "purpose": "customer_name|revenue_amount|date_filter|grouping",
          "data_type": "varchar|decimal|datetime"
        }}
      ]
    }}
  ],
  "sql_query": "SELECT TOP (10) [CustomerName], [TotalRevenue] FROM [dbo].[CustomerSummary] ORDER BY [TotalRevenue] DESC",
  "confidence": 0.9
}}

EXAMPLES OF GOOD SIMPLE SQL:
- Single table ranking: "SELECT TOP (10) [CustomerName], [Revenue] FROM [dbo].[Customers] ORDER BY [Revenue] DESC"
- Single table aggregation: "SELECT SUM([Amount]) AS TotalRevenue FROM [dbo].[Payments]"
- Single table with filter: "SELECT [Name], [Amount] FROM [dbo].[Contracts] WHERE [Status] = 'Active'"

IMPORTANT: 
- Only select tables that actually exist in the provided context
- Choose columns that make business sense for the question
- Use proper SQL Server syntax with square brackets
- For money/revenue, look for columns with 'amount', 'total', 'revenue', 'price' in name
- For customer names, look for columns with 'name', 'title', 'description'
- For time filters, use actual date columns, not geographic coordinates!
- KEEP IT SIMPLE - prefer single table queries when possible"""

            messages = [
                SystemMessage(content="You are a business intelligence expert. Analyze the database intelligently and respond with JSON only."),
                HumanMessage(content=prompt)
            ]
            
            response = await asyncio.to_thread(self.llm.invoke, messages)
            return parse_json_response(response.content)
            
        except Exception as e:
            print(f"      ⚠️ Intelligent analysis failed: {e}")
            return None
    
    def _create_fallback_response(self, question: str, tables: List[TableInfo]) -> Dict[str, Any]:
        """Create fallback response when LLM fails"""
        # Simple keyword-based fallback
        q_lower = question.lower()
        
        if 'customer' in q_lower and any(word in q_lower for word in ['revenue', 'paid', 'sales']):
            return {
                'intent_summary': 'Customer revenue analysis',
                'query_type': 'ranking',
                'selected_tables': [],
                'sql_query': '',
                'confidence': 0.3
            }
        
        return {
            'intent_summary': 'General query',
            'query_type': 'list',
            'selected_tables': [],
            'sql_query': '',
            'confidence': 0.1
        }

class SafetyValidator:
    """SQL safety validation with sqlglot integration"""
    
    def validate_sql_safety(self, sql: str) -> bool:
        """Enhanced SQL safety validation"""
        if not sql or len(sql.strip()) < 5:
            return False
        
        sql_normalized = ' ' + sql.upper().strip() + ' '
        
        # Basic safety checks
        safe_starts = [' SELECT ', ' WITH ']
        if not any(sql_normalized.startswith(start.strip()) for start in safe_starts):
            return False
        
        # Dangerous operations
        dangerous_patterns = [
            r'\b(?:INSERT|UPDATE|DELETE|MERGE|REPLACE)\b',
            r'\b(?:DROP|CREATE|ALTER|TRUNCATE)\b',
            r'\b(?:EXEC|EXECUTE|SP_|XP_|DBCC)\b',
            r'\b(?:GRANT|REVOKE|DENY)\b',
            r'\b(?:BULK|OPENROWSET|OPENDATASOURCE)\b',
            r'\bXP_CMDSHELL\b',
            r'\b(?:BACKUP|RESTORE)\b'
        ]
        
        import re
        for pattern in dangerous_patterns:
            if re.search(pattern, sql_normalized, re.IGNORECASE):
                return False
        
        # Check for multiple statements
        sql_clean = sql.strip().rstrip(';')
        if ';' in sql_clean:
            return False
        
        # sqlglot validation if available
        if HAS_SQLGLOT:
            try:
                parsed = sqlglot.parse_one(sql, dialect="tsql")
                if not parsed:
                    return False
                
                if not isinstance(parsed, sqlglot.expressions.Select):
                    return False
                
                # Check for dangerous subqueries
                for node in parsed.walk():
                    if isinstance(node, (sqlglot.expressions.Insert, 
                                       sqlglot.expressions.Update, 
                                       sqlglot.expressions.Delete)):
                        return False
                
                return True
                
            except Exception:
                return False
        
        return True

class QueryExecutor:
    """Query executor with enhanced error handling"""
    
    def __init__(self, config: Config):
        self.config = config
    
    async def execute_query(self, sql: str):
        """Execute SQL with enhanced error handling"""
        print("   🔄 Executing query...")
        print(f"      📝 SQL: {sql[:80]}{'...' if len(sql) > 80 else ''}")
        
        try:
            with pyodbc.connect(self.config.get_database_connection_string()) as conn:
                if self.config.utf8_encoding:
                    conn.setdecoding(pyodbc.SQL_CHAR, encoding='utf-8')
                    conn.setdecoding(pyodbc.SQL_WCHAR, encoding='utf-8')
                    conn.setencoding(encoding='utf-8')
                
                cursor = conn.cursor()
                cursor.execute(sql)
                
                if cursor.description:
                    columns = [col[0] for col in cursor.description]
                    results = []
                    
                    for row in cursor:
                        row_dict = {}
                        for i, value in enumerate(row):
                            if i < len(columns):
                                row_dict[columns[i]] = safe_database_value(value)
                        results.append(row_dict)
                    
                    print(f"      ✅ Query executed: {len(results)} rows")
                    return results, None
                else:
                    return [], None
                    
        except Exception as e:
            error_msg = str(e)
            print(f"      ❌ Query failed: {error_msg}")
            
            # Enhanced error suggestions
            if "Incorrect syntax" in error_msg:
                error_msg += " | Try simpler column names or check table structure"
            elif "Invalid column name" in error_msg:
                error_msg += " | Column may not exist in selected table"
            elif "Invalid object name" in error_msg:
                error_msg += " | Table may not exist or wrong schema"
            elif "Arithmetic overflow" in error_msg:
                error_msg += " | Try using different numeric columns"
            
            return [], error_msg

class QueryInterface:
    """Enhanced Query Interface with LLM-driven intelligence"""
    
    def __init__(self, config: Config):
        self.config = config
        
        # Initialize LLM
        self.llm = AzureChatOpenAI(
            azure_endpoint=config.azure_endpoint,
            api_key=config.api_key,
            azure_deployment=config.deployment_name,
            api_version=config.api_version,
            request_timeout=90,  # Increased for intelligent analysis
            # temperature=0.1,     # Low temperature for consistent results
        )
        
        # Initialize components
        self.analyzer = IntelligentAnalyzer(self.llm)
        self.safety_validator = SafetyValidator()
        self.executor = QueryExecutor(config)
        
        print("✅ Intelligent Query Interface initialized")
        print("   🧠 LLM-driven table and column selection")
        print("   🎯 Dynamic business context understanding")
        print(f"   🛡️ Safety validation: {'✅ sqlglot' if HAS_SQLGLOT else '⚠️ basic only'}")
    
    async def start_session(self, tables: List[TableInfo], domain: Optional[BusinessDomain], relationships: List[Relationship]):
        """Start intelligent session"""
        
        # Show system readiness
        self._show_system_readiness(tables, domain)
        
        query_count = 0
        
        while True:
            try:
                question = input(f"\n❓ Query #{query_count + 1}: ").strip()
                
                if question.lower() in ['quit', 'exit', 'q']:
                    break
                if not question:
                    continue
                
                query_count += 1
                print(f"🔄 Processing...")
                
                start_time = time.time()
                result = await self.process_query(question, tables)
                result.execution_time = time.time() - start_time
                
                self._display_result(result)
                
            except KeyboardInterrupt:
                print("\n⏸️ Interrupted")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
    
    async def process_query(self, question: str, tables: List[TableInfo]) -> QueryResult:
        """Process query using intelligent LLM-driven approach"""
        
        try:
            # Single intelligent analysis
            analysis = await self.analyzer.analyze_and_generate_sql(question, tables)
            
            if not analysis or analysis.get('confidence', 0) < 0.5:
                return QueryResult(
                    question=question,
                    sql_query="",
                    results=[],
                    error="Could not understand the question. Please try rephrasing with specific business terms.",
                    result_type="error"
                )
            
            # Extract SQL
            sql = analysis.get('sql_query', '').strip()
            
            if not sql:
                return QueryResult(
                    question=question,
                    sql_query="",
                    results=[],
                    error="Could not generate SQL. Try asking about specific data like 'customers', 'revenue', or 'sales'.",
                    result_type="error"
                )
            
            # Safety validation
            if not self.safety_validator.validate_sql_safety(sql):
                return QueryResult(
                    question=question,
                    sql_query=sql,
                    results=[],
                    error="Generated SQL failed safety validation. Try a simpler query.",
                    result_type="error"
                )
            
            # Execute validated SQL
            results, error = await self.executor.execute_query(sql)
            
            # Extract table names from analysis
            tables_used = []
            for table_info in analysis.get('selected_tables', []):
                table_name = table_info.get('table_name', '')
                if table_name:
                    tables_used.append(table_name)
            
            return QueryResult(
                question=question,
                sql_query=sql,
                results=results,
                error=error,
                tables_used=tables_used,
                result_type="data" if results and not error else "error",
                sql_generation_method="intelligent_llm",
                intent_confidence=analysis.get('confidence', 0.8)
            )
            
        except Exception as e:
            return QueryResult(
                question=question,
                sql_query="",
                results=[],
                error=f"Query processing failed: {str(e)}",
                result_type="error"
            )
    
    def _show_system_readiness(self, tables: List[TableInfo], domain: Optional[BusinessDomain]):
        """Show intelligent system readiness"""
        
        # Analyze available data intelligently
        business_areas = {
            'Customer Data': 0,
            'Financial Data': 0, 
            'Sales Data': 0,
            'Employee Data': 0,
            'Contract Data': 0,
            'Other Data': 0
        }
        
        for table in tables:
            name_lower = table.name.lower()
            
            if any(word in name_lower for word in ['customer', 'client', 'account', 'contact']):
                business_areas['Customer Data'] += 1
            elif any(word in name_lower for word in ['payment', 'transaction', 'invoice', 'billing', 'revenue']):
                business_areas['Financial Data'] += 1
            elif any(word in name_lower for word in ['sales', 'deal', 'opportunity', 'order']):
                business_areas['Sales Data'] += 1
            elif any(word in name_lower for word in ['user', 'employee', 'staff', 'rep']):
                business_areas['Employee Data'] += 1
            elif any(word in name_lower for word in ['contract', 'agreement']):
                business_areas['Contract Data'] += 1
            else:
                business_areas['Other Data'] += 1
        
        print(f"\n🚀 INTELLIGENT QUERY SYSTEM READY:")
        print(f"   📊 Total tables: {len(tables)}")
        
        for area, count in business_areas.items():
            if count > 0:
                emoji = "🔥" if area in ['Customer Data', 'Financial Data', 'Sales Data'] else "📋"
                print(f"   {emoji} {area}: {count} tables")
        
        print(f"\n🧠 INTELLIGENT CAPABILITIES:")
        print(f"   • Natural language understanding")
        print(f"   • Dynamic table and column selection")
        print(f"   • Business context awareness")
        print(f"   • Safe SQL generation")
        
        print(f"\n💡 Try asking:")
        print(f"   • Who are our top customers by revenue?")
        print(f"   • Which sales reps closed the most deals?")
        print(f"   • What's our total revenue this year?")
        print(f"   • Show me contract details")
        print(f"   • List payment transactions")
    
    def _display_result(self, result: QueryResult):
        """Display query results"""
        
        print(f"⏱️ Completed in {result.execution_time:.1f}s")
        print("-" * 50)
        
        if result.is_successful() and result.has_results():
            print(f"✅ QUERY SUCCESSFUL")
            print(f"\n📋 Generated SQL ({result.sql_generation_method}):")
            print(f"{result.sql_query}")
            
            if result.tables_used:
                # Clean table names for display
                clean_names = []
                for table in result.tables_used:
                    clean_name = table.replace('[', '').replace(']', '').split('.')[-1]
                    clean_names.append(clean_name)
                print(f"\n📊 Tables Used: {', '.join(clean_names)}")
            
            print(f"\n📈 Results ({len(result.results)} rows):")
            self._display_data(result.results)
            
        else:
            print(f"❌ QUERY FAILED")
            print(f"   Error: {result.error}")
            
            print(f"\n💡 Enhanced suggestions:")
            print(f"   • Try: 'top 10 customers by revenue'")
            print(f"   • Try: 'which sales reps closed most deals'")
            print(f"   • Try: 'total contract value this year'")
    
    def _display_data(self, results: List[Dict[str, Any]]):
        """Enhanced data display with business formatting"""
        if len(results) == 1 and len(results[0]) == 1:
            # Single value result
            key, value = next(iter(results[0].items()))
            
            # Format based on column name
            if any(word in key.lower() for word in ['revenue', 'amount', 'total', 'value']):
                if isinstance(value, (int, float)) and abs(value) >= 1000:
                    formatted_value = f"${value:,.2f}"
                else:
                    formatted_value = f"${value}"
            elif any(word in key.lower() for word in ['count', 'number', 'qty']):
                formatted_value = f"{value:,}" if isinstance(value, (int, float)) else str(value)
            else:
                formatted_value = str(value)
            
            print(f"   🎯 {key}: {formatted_value}")
            
        else:
            # Multiple rows - enhanced business formatting
            for i, row in enumerate(results[:15], 1):
                parts = []
                for j, (key, value) in enumerate(row.items()):
                    if j >= 4:  # Limit columns
                        break
                    
                    # Enhanced business formatting
                    if isinstance(value, (int, float)):
                        if any(word in key.lower() for word in ['revenue', 'amount', 'total', 'value', 'price']):
                            formatted_value = f"${value:,.2f}" if abs(value) >= 1000 else f"${value}"
                        elif any(word in key.lower() for word in ['count', 'number', 'qty']):
                            formatted_value = f"{value:,}"
                        else:
                            formatted_value = f"{value:,}" if abs(value) >= 1000 else str(value)
                    elif isinstance(value, str):
                        formatted_value = value[:40] + "..." if len(value) > 40 else value
                    else:
                        formatted_value = str(value) if value is not None else ""
                    
                    # Enhanced key formatting
                    if any(word in key.lower() for word in ['revenue', 'amount', 'total']):
                        display_key = "Revenue"
                    elif any(word in key.lower() for word in ['name', 'title']):
                        display_key = key.replace('_', ' ').title()
                    elif any(word in key.lower() for word in ['count', 'number']):
                        display_key = "Count"
                    else:
                        display_key = key.replace('_', ' ').title()
                    
                    parts.append(f"{display_key}: {formatted_value}")
                
                print(f"   {i:2d}. {' | '.join(parts)}")
            
            if len(results) > 15:
                print(f"   ... and {len(results) - 15} more rows")