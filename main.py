#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Semantic Database RAG System - Enhanced Main Entry Point
Simple, Readable, Maintainable - DRY, SOLID, YAGNI principles
Function names: SemanticRAG (not SimplifiedSemanticRAG)
FIXED: Only use discovered columns, enhanced safety validation
"""

import asyncio
import os
import sys
from pathlib import Path

def load_env():
    """Load environment variables from .env file"""
    env_file = Path('.env')
    if env_file.exists():
        with open(env_file, 'r', encoding='utf-8') as f:
            for line in f:
                if '=' in line and not line.startswith('#'):
                    key, value = line.strip().split('=', 1)
                    os.environ[key] = value.strip('"\'')

def check_dependencies() -> bool:
    """Check required dependencies"""
    missing_deps = []
    
    try:
        import pyodbc
        print("✅ pyodbc: SQL Server connectivity")
    except ImportError:
        missing_deps.append("pyodbc")
    
    try:
        from langchain_openai import AzureChatOpenAI
        print("✅ langchain-openai: LLM integration")
    except ImportError:
        missing_deps.append("langchain-openai")
    
    try:
        import sqlglot
        print("✅ sqlglot: SQL parsing and validation")
    except ImportError:
        print("⚠️ sqlglot: Not available - using basic SQL validation")
    
    if missing_deps:
        print(f"❌ Missing critical dependencies: {', '.join(missing_deps)}")
        print("💡 Install with: pip install pyodbc langchain-openai sqlglot")
        return False
    
    return True

class SemanticRAG:
    """Enhanced SemanticRAG system with discovered-columns-only approach"""
    
    def __init__(self):
        load_env()
        self._initialize_system()
    
    def _initialize_system(self):
        """Initialize system components"""
        try:
            if not check_dependencies():
                raise RuntimeError("Missing critical dependencies")
            
            from shared.config import Config
            self.config = Config()
            
            health = self.config.get_health_check()
            if not health['llm_configured'] or not health['database_configured']:
                missing = []
                if not health['llm_configured']:
                    missing.append("Azure OpenAI (AZURE_OPENAI_API_KEY, AZURE_ENDPOINT)")
                if not health['database_configured']:
                    missing.append("SQL Server (DATABASE_CONNECTION_STRING)")
                raise RuntimeError(f"Configuration incomplete: {', '.join(missing)}")
            
            print("🔧 Initializing enhanced system components...")
            
            sys.path.insert(0, str(Path(__file__).parent))
            
            from db.discovery import DatabaseDiscovery
            self.discovery = DatabaseDiscovery(self.config)
            
            # Import the enhanced semantic analyzer
            from semantic.analysis import SemanticAnalyzer
            self.analyzer = SemanticAnalyzer(self.config)
            
            # Import the fixed query interface
            from interactive.query_interface import QueryInterface
            self.query_interface = QueryInterface(self.config)
            
            print("✅ Enhanced SemanticRAG System initialized")
            print("   🎯 Discovered-columns-only approach enabled")
            print("   📊 Smart table selection with column validation")
            print("   🛡️ Fixed safety validation with CTE support")
            print("   🔍 Enhanced column discovery and classification")
            
        except Exception as e:
            print(f"❌ Initialization failed: {e}")
            print("\n🔧 Setup Checklist:")
            print("   1. Copy env_example.txt to .env")
            print("   2. Set AZURE_OPENAI_API_KEY")
            print("   3. Set DATABASE_CONNECTION_STRING")
            print("   4. Set AZURE_ENDPOINT")
            print("   5. Install dependencies: pip install pyodbc langchain-openai sqlglot")
            raise
    
    async def run_discovery(self) -> bool:
        """Run database discovery with enhanced column analysis"""
        print("\n🔍 ENHANCED SQL SERVER DISCOVERY WITH RDL INTEGRATION")
        print("=" * 60)
        
        try:
            success = await self.discovery.discover_database()
            
            if success:
                stats = self.discovery.get_discovery_stats()
                print(f"✅ Discovery completed!")
                print(f"   📊 Tables: {stats['tables']}")
                print(f"   👁️ Views: {stats['views']}")
                print(f"   🔗 Relationships: {stats['relationships']}")
                print(f"   📋 RDL references: {stats['rdl_references']}")
                print(f"   🏷️ RDL fields: {stats['rdl_fields']}")
                print(f"   🔄 RDL JOIN pairs: {stats['rdl_join_pairs']}")
                return True
            else:
                print("❌ Discovery failed")
                return False
                
        except Exception as e:
            print(f"❌ Discovery error: {e}")
            return False
    
    async def run_analysis(self) -> bool:
        """Run enhanced semantic analysis with column discovery"""
        print("\n🧠 ENHANCED SEMANTIC ANALYSIS WITH COLUMN DISCOVERY")
        print("=" * 55)
        
        try:
            tables = self.discovery.get_tables()
            if not tables:
                if self.discovery.load_from_cache():
                    tables = self.discovery.get_tables()
                else:
                    print("❌ No tables found. Run Discovery first.")
                    return False
            
            success = await self.analyzer.analyze_tables(tables, self.discovery)
            
            if success:
                stats = self.analyzer.get_analysis_stats()
                print("✅ Enhanced column-discovery analysis completed!")
                print(f"   📊 Total objects: {stats['total_objects']}")
                print(f"   💰 Revenue-ready: {stats['revenue_ready_tables']}")
                print(f"   👥 Customer tables: {stats['customer_tables']}")
                print(f"   💳 Payment tables: {stats['payment_tables']}")
                print(f"   🔢 Discovered measures: {stats['total_discovered_measures']}")
                
                if stats['revenue_analytics_enabled']:
                    print("   🔥 Revenue analytics enabled!")
                
                return True
            else:
                print("❌ Analysis failed")
                return False
                
        except Exception as e:
            print(f"❌ Analysis error: {e}")
            return False
    
    async def run_queries(self) -> bool:
        """Run enhanced query interface with discovered-columns-only approach"""
        print("\n💬 ENHANCED QUERY INTERFACE - DISCOVERED COLUMNS ONLY")
        print("=" * 55)
        
        try:
            tables = []
            domain = None
            relationships = []
            
            if hasattr(self.analyzer, 'get_tables'):
                tables = self.analyzer.get_tables()
                domain = self.analyzer.get_domain()
                relationships = self.analyzer.get_relationships()
            
            if not tables:
                print("   🔄 Loading from caches...")
                
                if self.analyzer.load_from_cache():
                    tables = self.analyzer.get_tables()
                    domain = self.analyzer.get_domain()
                    relationships = self.analyzer.get_relationships()
                elif self.discovery.load_from_cache():
                    tables = self.discovery.get_tables()
                    relationships = self.discovery.get_relationships()
            
            if not tables:
                print("❌ No data available. Please run:")
                print("   1. Discovery")
                print("   2. Analysis")
                return False
            
            # Validate tables have discovered columns
            valid_tables = [t for t in tables if t.columns]
            invalid_tables = len(tables) - len(valid_tables)
            
            if invalid_tables > 0:
                print(f"   ⚠️ {invalid_tables} tables have no discovered columns")
            
            if not valid_tables:
                print("❌ No tables with discovered columns found.")
                print("   Please run Discovery and Analysis again.")
                return False
            
            # Show enhanced capabilities
            revenue_ready = len([t for t in valid_tables if getattr(t, 'revenue_readiness', 0) >= 0.7])
            customer_tables = len([t for t in valid_tables if t.entity_type == 'Customer'])
            payment_tables = len([t for t in valid_tables if t.entity_type in ['Payment', 'CustomerRevenue']])
            total_measures = sum(len(getattr(t, 'measures', [])) for t in valid_tables)
            total_names = sum(len(getattr(t, 'name_columns', [])) for t in valid_tables)
            
            print(f"🚀 Enhanced query interface ready:")
            print(f"   📊 Valid tables: {len(valid_tables)}")
            print(f"   💰 Revenue-ready tables: {revenue_ready}")
            print(f"   👥 Customer tables: {customer_tables}")
            print(f"   💳 Payment tables: {payment_tables}")
            print(f"   🔢 Total discovered measures: {total_measures}")
            print(f"   🏷️ Total discovered names: {total_names}")
            
            if revenue_ready > 0:
                print(f"   ✅ Revenue queries supported!")
            
            if domain and domain.sample_questions:
                print(f"\n💡 Try asking:")
                for i, question in enumerate(domain.sample_questions[:4], 1):
                    print(f"   {i}. {question}")
            
            # Use valid tables for querying
            await self.query_interface.start_session(valid_tables, domain, relationships)
            return True
            
        except Exception as e:
            print(f"❌ Query interface error: {e}")
            return False
    
    def show_status(self) -> None:
        """Show enhanced system status"""
        print("\n📊 ENHANCED SYSTEM STATUS")
        print("=" * 30)
        
        try:
            discovery_tables = self.discovery.get_tables()
            if not discovery_tables:
                self.discovery.load_from_cache()
                discovery_tables = self.discovery.get_tables()
            
            analyzer_tables = []
            if hasattr(self.analyzer, 'load_from_cache'):
                self.analyzer.load_from_cache()
                analyzer_tables = self.analyzer.get_tables()
            
            print(f"🔍 Discovery: {'✅ Ready' if discovery_tables else '❌ Incomplete'}")
            if discovery_tables:
                stats = self.discovery.get_discovery_stats()
                valid_tables = [t for t in discovery_tables if t.columns]
                print(f"   📊 Objects: {stats['total_objects']}")
                print(f"   ✅ With columns: {len(valid_tables)}")
                print(f"   📋 RDL references: {stats['rdl_references']}")
                print(f"   🏷️ RDL fields: {stats['rdl_fields']}")
            
            print(f"🧠 Analysis: {'✅ Ready' if analyzer_tables else '❌ Incomplete'}")
            if analyzer_tables:
                stats = self.analyzer.get_analysis_stats()
                print(f"   📊 Objects: {stats['total_objects']}")
                print(f"   💰 Revenue-ready: {stats['revenue_ready_tables']}")
                print(f"   🔢 Discovered measures: {stats['total_discovered_measures']}")
                
                if stats['revenue_analytics_enabled']:
                    print(f"   🔥 Revenue analytics: Enabled")
            
            print(f"💬 Queries: {'✅ Ready' if (analyzer_tables or discovery_tables) else '❌ Not ready'}")
            
            health = self.config.get_health_check()
            print(f"\n🏥 Health:")
            print(f"   🔑 LLM: {'✅' if health['llm_configured'] else '❌'}")
            print(f"   🗄️ Database: {'✅' if health['database_configured'] else '❌'}")
            print(f"   💾 Cache: {'✅' if health['cache_writable'] else '❌'}")
            
            # Enhanced validation
            if discovery_tables:
                tables_with_columns = [t for t in discovery_tables if t.columns]
                print(f"   📊 Tables with columns: {len(tables_with_columns)}/{len(discovery_tables)}")
                
                if analyzer_tables:
                    tables_with_measures = [t for t in analyzer_tables if getattr(t, 'measures', [])]
                    tables_with_names = [t for t in analyzer_tables if getattr(t, 'name_columns', [])]
                    print(f"   🔢 Tables with measures: {len(tables_with_measures)}")
                    print(f"   🏷️ Tables with names: {len(tables_with_names)}")
            
        except Exception as e:
            print(f"⚠️ Status check error: {e}")

def show_menu() -> None:
    """Display enhanced main menu"""
    print("\n" + "="*70)
    print("ENHANCED SEMANTIC DATABASE RAG SYSTEM")
    print("Discovered-Columns-Only Revenue-Focused Table Selection")
    print("="*70)
    print("1. 🔍 SQL Server Discovery")
    print("   • Discover tables, views, and columns")
    print("   • Extract RDL business context and field mappings")
    print("   • Build enhanced relationship graph with JOIN pairs")
    print()
    print("2. 🧠 Enhanced Semantic Analysis")
    print("   • Column discovery and classification")
    print("   • Revenue table prioritization")
    print("   • Only use discovered columns for queries")
    print()
    print("3. 💬 Enhanced Query Interface")
    print("   • Discovered-columns-only SQL generation")
    print("   • Fixed safety validation with CTE support")
    print("   • Smart table selection with column validation")
    print()
    print("4. 📊 Enhanced System Status")
    print("0. Exit")
    print("="*70)

async def handle_choice(system: SemanticRAG, choice: str) -> bool:
    """Handle menu choice"""
    try:
        if choice == '1':
            return await system.run_discovery()
        elif choice == '2':
            return await system.run_analysis()
        elif choice == '3':
            return await system.run_queries()
        elif choice == '4':
            system.show_status()
            return True
        else:
            print(f"❌ Invalid choice: '{choice}'. Please enter 0-4.")
            return True
    except KeyboardInterrupt:
        print("\n⏸️ Operation interrupted")
        return True
    except Exception as e:
        print(f"❌ Operation failed: {e}")
        return True

def main():
    """Enhanced main entry point"""
    print("🚀 ENHANCED SEMANTIC DATABASE RAG SYSTEM")
    print("Discovered-Columns-Only Revenue-Focused Table Selection")
    print("=" * 70)
    
    try:
        system = SemanticRAG()
    except Exception:
        print("\n❌ System initialization failed")
        return
    
    while True:
        show_menu()
        
        try:
            choice = input("Enter choice (0-4): ").strip()
            
            if choice == '0':
                print("👋 Thanks for using Enhanced Semantic Database RAG!")
                break
            
            asyncio.run(handle_choice(system, choice))
            
        except KeyboardInterrupt:
            print("\n⏸️ Interrupted")
            break
        except Exception as e:
            print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()