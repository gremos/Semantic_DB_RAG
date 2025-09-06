#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Enhanced Semantic Database RAG System - SQL Server Architecture
Architecture: SQL Server sys.* metadata, RDL integration, sqlglot validation, cross-industry entities
Simple, Readable, Maintainable - DRY, SOLID, YAGNI principles
Function names: SemanticRAG (not SimplifiedSemanticRAG)
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
    """Check required dependencies for enhanced architecture"""
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
        print("⚠️ sqlglot: Not available - SQL safety validation will be basic only")
        print("   Install with: pip install sqlglot")
    
    if missing_deps:
        print(f"❌ Missing critical dependencies: {', '.join(missing_deps)}")
        print("💡 Install with: pip install pyodbc langchain-openai sqlglot")
        return False
    
    return True

class SemanticRAG:
    """Enhanced semantic RAG system with SQL Server architecture integration"""
    
    def __init__(self):
        load_env()
        self._initialize_enhanced_system()
    
    def _initialize_enhanced_system(self):
        """Initialize enhanced system with architecture components"""
        try:
            # Check dependencies first
            if not check_dependencies():
                raise RuntimeError("Missing critical dependencies")
            
            # Load enhanced configuration
            from shared.config import Config
            self.config = Config()
            
            # Validate enhanced configuration
            health = self.config.get_health_check()
            if not health['llm_configured'] or not health['database_configured']:
                missing = []
                if not health['llm_configured']:
                    missing.append("Azure OpenAI (AZURE_OPENAI_API_KEY, AZURE_ENDPOINT)")
                if not health['database_configured']:
                    missing.append("SQL Server (DATABASE_CONNECTION_STRING)")
                raise RuntimeError(f"Configuration incomplete: {', '.join(missing)}")
            
            # Initialize enhanced components
            print("🔧 Initializing enhanced architecture components...")
            
            # Import enhanced modules
            sys.path.insert(0, str(Path(__file__).parent))
            
            # Enhanced discovery with SQL Server sys.* and RDL
            from db.discovery import DatabaseDiscovery
            self.discovery = DatabaseDiscovery(self.config)
            
            # Enhanced semantic analysis with cross-industry entities
            from semantic.analysis import SemanticAnalyzer
            self.analyzer = SemanticAnalyzer(self.config)
            
            # Enhanced query interface with fixed templates
            from interactive.query_interface import QueryInterface
            self.query_interface = QueryInterface(self.config)
            
            print("✅ Enhanced SemanticRAG System initialized")
            print("   🏢 Architecture: SQL Server sys.* metadata + RDL integration")
            print("   🌐 Entities: Cross-industry taxonomy (Customer, Contract, Employee, etc.)")
            print("   ⚡ SQL Generation: Fixed templates with LLM fallback")
            print("   🛡️ Safety: sqlglot validation for SQL injection prevention")
            
        except Exception as e:
            print(f"❌ Enhanced initialization failed: {e}")
            print("\n🔧 Architecture Setup Checklist:")
            print("   1. Copy env_example.txt to .env")
            print("   2. Set AZURE_OPENAI_API_KEY with your Azure OpenAI API key")
            print("   3. Set DATABASE_CONNECTION_STRING for SQL Server")
            print("   4. Set AZURE_ENDPOINT with your Azure OpenAI endpoint")
            print("   5. Install dependencies: pip install pyodbc langchain-openai sqlglot")
            raise
    
    async def run_enhanced_discovery(self) -> bool:
        """Run enhanced database discovery with SQL Server sys.* and RDL integration"""
        print("\n🔍 ENHANCED SQL SERVER DISCOVERY + RDL INTEGRATION")
        print("Architecture: sys.* metadata + sqlglot parsing + RDL business context")
        print("=" * 70)
        
        try:
            success = await self.discovery.discover_database()
            
            if success:
                stats = self.discovery.get_discovery_stats()
                print(f"✅ Enhanced discovery completed!")
                print(f"   📊 Tables: {stats['tables']}")
                print(f"   👁️ Views: {stats['views']} (with definitions)")
                print(f"   🔗 Relationships: {stats['relationships']} (from foreign keys)")
                print(f"   📋 RDL references: {stats['rdl_references']}")
                print(f"   🗃️ Metadata source: {stats['metadata_source']}")
                print(f"   📝 Sampling method: {stats['sampling_method']}")
                print(f"   ⚙️ SQL parsing: {'✅ sqlglot' if stats['sqlglot_available'] else '⚠️ basic only'}")
                return True
            else:
                print("❌ Enhanced discovery failed")
                return False
                
        except Exception as e:
            print(f"❌ Discovery error: {e}")
            return False
    
    async def run_enhanced_analysis(self) -> bool:
        """Run enhanced semantic analysis with cross-industry entities"""
        print("\n🧠 ENHANCED CROSS-INDUSTRY SEMANTIC ANALYSIS")
        print("Architecture: Cross-industry entity taxonomy + RDL business context")
        print("=" * 65)
        
        try:
            # Get tables from enhanced discovery
            tables = self.discovery.get_tables()
            if not tables:
                if self.discovery.load_from_cache():
                    tables = self.discovery.get_tables()
                else:
                    print("❌ No tables found. Run Enhanced Discovery first.")
                    return False
            
            # Run enhanced semantic analysis with RDL integration
            success = await self.analyzer.analyze_tables(tables, self.discovery)
            
            if success:
                stats = self.analyzer.get_analysis_stats()
                print("✅ Enhanced cross-industry analysis completed!")
                print(f"   📊 Total objects: {stats['total_objects']}")
                print(f"   📋 Tables: {stats['total_tables']}")
                print(f"   👁️ Views: {stats['total_views']}")
                print(f"   🌐 Entity types: {stats['entity_types_found']}")
                print(f"   🔥 High priority: {stats['high_priority_tables']}")
                print(f"   📈 Fact tables: {stats['fact_tables']}")
                print(f"   🏢 Domain: {stats['domain_type']}")
                print(f"   🏭 Industry: {stats['industry']}")
                print(f"   🧠 Analysis method: {stats['analysis_method']}")
                
                # Show entity breakdown
                entity_counts = [
                    ('Customer', stats.get('customer_tables', 0)),
                    ('Payment', stats.get('payment_tables', 0)),
                    ('Contract', stats.get('contract_tables', 0)),
                    ('Employee', stats.get('employee_tables', 0))
                ]
                print(f"   🎯 Key entities:")
                for entity, count in entity_counts:
                    if count > 0:
                        print(f"      • {entity}: {count}")
                
                return True
            else:
                print("❌ Enhanced analysis failed")
                return False
                
        except Exception as e:
            print(f"❌ Analysis error: {e}")
            return False
    
    async def run_enhanced_queries(self) -> bool:
        """Run enhanced query interface with fixed templates and safety validation"""
        print("\n💬 ENHANCED QUERY INTERFACE")
        print("Architecture: Fixed SQL templates + LLM fallback + sqlglot safety validation")
        print("=" * 75)
        
        try:
            # Load enhanced data
            tables = []
            domain = None
            relationships = []
            
            # Try enhanced analyzer first (has enriched tables with entity types)
            if hasattr(self.analyzer, 'get_tables'):
                tables = self.analyzer.get_tables()
                domain = self.analyzer.get_domain()
                relationships = self.analyzer.get_relationships()
            
            # Fallback to caches
            if not tables:
                print("   🔄 Loading from enhanced caches...")
                
                if self.analyzer.load_from_cache():
                    tables = self.analyzer.get_tables()
                    domain = self.analyzer.get_domain()
                    relationships = self.analyzer.get_relationships()
                elif self.discovery.load_from_cache():
                    tables = self.discovery.get_tables()
                    relationships = self.discovery.get_relationships()
            
            if not tables:
                print("❌ No enhanced data available. Please run:")
                print("   1. Enhanced SQL Server Discovery")
                print("   2. Enhanced Cross-Industry Analysis")
                return False
            
            # Show enhanced readiness with cross-industry entities
            entities = {}
            for table in tables:
                entity = table.entity_type
                entities[entity] = entities.get(entity, 0) + 1
            
            print(f"🚀 Enhanced query interface ready:")
            print(f"   📊 Total objects: {len(tables)}")
            print(f"   🌐 Entity types: {len(entities)}")
            
            # Show top cross-industry entities
            sorted_entities = sorted(entities.items(), key=lambda x: x[1], reverse=True)[:8]
            priority_entities = ['Customer', 'Payment', 'Contract', 'Order', 'Employee']
            
            for entity, count in sorted_entities:
                priority_emoji = "🔥" if entity in priority_entities else "📋"
                print(f"   {priority_emoji} {entity}: {count} objects")
            
            print(f"   🔄 Enhanced Pipeline:")
            print(f"      1. Cross-Industry Intent Analysis")
            print(f"      2. Smart Multi-Factor Table Selection")
            print(f"      3. Fixed SQL Templates + Safety Validation")
            
            # Show domain-specific sample questions
            if domain and domain.sample_questions:
                print(f"\n💡 Sample questions for {domain.domain_type}:")
                for i, question in enumerate(domain.sample_questions[:4], 1):
                    print(f"   {i}. {question}")
            
            await self.query_interface.start_session(tables, domain, relationships)
            return True
            
        except Exception as e:
            print(f"❌ Enhanced query interface error: {e}")
            return False
    
    def show_enhanced_status(self) -> None:
        """Show enhanced system status with architecture details"""
        print("\n📊 ENHANCED SYSTEM STATUS")
        print("Architecture: SQL Server + RDL + Cross-Industry + Safety")
        print("=" * 55)
        
        try:
            # Check enhanced discovery
            discovery_tables = self.discovery.get_tables()
            if not discovery_tables:
                self.discovery.load_from_cache()
                discovery_tables = self.discovery.get_tables()
            
            # Check enhanced analysis
            analyzer_tables = []
            if hasattr(self.analyzer, 'load_from_cache'):
                self.analyzer.load_from_cache()
                analyzer_tables = self.analyzer.get_tables()
            
            # Enhanced discovery status
            print(f"🔍 Enhanced Discovery: {'✅ Ready' if discovery_tables else '❌ Incomplete'}")
            if discovery_tables:
                stats = self.discovery.get_discovery_stats()
                print(f"   📊 Objects: {stats['total_objects']}")
                print(f"   📋 Tables: {stats['tables']}")
                print(f"   👁️ Views: {stats['views']}")
                print(f"   🔗 Relationships: {stats['relationships']}")
                print(f"   📋 RDL references: {stats['rdl_references']}")
                print(f"   🗃️ Metadata: {stats['metadata_source']}")
                print(f"   ⚙️ SQL parsing: {'✅ sqlglot' if stats['sqlglot_available'] else '⚠️ basic'}")
            
            # Enhanced analysis status
            print(f"🧠 Enhanced Analysis: {'✅ Ready' if analyzer_tables else '❌ Incomplete'}")
            if analyzer_tables:
                stats = self.analyzer.get_analysis_stats()
                print(f"   📊 Objects: {stats['total_objects']}")
                print(f"   🌐 Entity types: {stats['entity_types_found']}")
                print(f"   🔥 High priority: {stats.get('high_priority_tables', 0)}")
                print(f"   📈 Fact tables: {stats.get('fact_tables', 0)}")
                print(f"   🏢 Domain: {stats.get('domain_type', 'Unknown')}")
                print(f"   🏭 Industry: {stats.get('industry', 'Unknown')}")
                print(f"   🧠 Method: {stats.get('analysis_method', 'unknown')}")
            
            # Query interface status
            print(f"💬 Enhanced Queries: {'✅ Ready' if (analyzer_tables or discovery_tables) else '❌ Not ready'}")
            
            # Show available cross-industry entities
            if analyzer_tables:
                entities = {}
                for table in analyzer_tables:
                    entity = table.entity_type
                    entities[entity] = entities.get(entity, 0) + 1
                
                print(f"\n🌐 Available Cross-Industry Entities:")
                sorted_entities = sorted(entities.items(), key=lambda x: x[1], reverse=True)
                priority_entities = ['Customer', 'Payment', 'Contract', 'Order', 'Employee', 'Product']
                
                for entity, count in sorted_entities[:12]:
                    priority_emoji = "🔥" if entity in priority_entities else "📋"
                    print(f"   {priority_emoji} {entity}: {count}")
            
            # System health check
            health = self.config.get_health_check()
            print(f"\n🏥 System Health:")
            print(f"   🔑 LLM configured: {'✅' if health['llm_configured'] else '❌'}")
            print(f"   🗄️ Database configured: {'✅' if health['database_configured'] else '❌'}")
            print(f"   💾 Cache writable: {'✅' if health['cache_writable'] else '❌'}")
            print(f"   🌍 UTF-8 support: {'✅' if health['utf8_support'] else '❌'}")
            print(f"   🎯 Overall health: {'✅ Excellent' if health['overall_health'] else '⚠️ Issues detected'}")
            
        except Exception as e:
            print(f"⚠️ Status check error: {e}")

def show_enhanced_menu() -> None:
    """Display enhanced main menu with architecture highlights"""
    print("\n" + "="*70)
    print("ENHANCED SEMANTIC DATABASE RAG SYSTEM")
    print("SQL Server Architecture + Cross-Industry + Safety")
    print("="*70)
    print("1. 🔍 Enhanced SQL Server Discovery")
    print("   • SQL Server sys.* metadata (not INFORMATION_SCHEMA)")
    print("   • RDL business context integration")
    print("   • First 3 + Last 3 sampling")
    print("   • View definitions with sqlglot parsing")
    print()
    print("2. 🧠 Enhanced Cross-Industry Analysis")
    print("   • Cross-industry entity taxonomy")
    print("   • Customer, Contract, Employee, Project, etc.")
    print("   • RDL business priority boosting")
    print("   • Deep column analysis (measures, names, keys)")
    print()
    print("3. 💬 Enhanced Query Interface")
    print("   • Fixed SQL templates with LLM fallback")
    print("   • sqlglot safety validation")
    print("   • Smart multi-factor table selection")
    print("   • Cross-industry intent analysis")
    print()
    print("4. 📊 Enhanced System Status")
    print("0. Exit")
    print("="*70)
    print("🏗️ Architecture Highlights:")
    print("   • SQL Server only (sys.* views, T-SQL dialect)")
    print("   • RDL integration for business context")
    print("   • sqlglot validation for SQL safety")
    print("   • Cross-industry entity support")
    print("   • Fixed templates over hallucinated SQL")

async def handle_enhanced_choice(system: SemanticRAG, choice: str) -> bool:
    """Handle menu choice with enhanced error handling"""
    try:
        if choice == '1':
            return await system.run_enhanced_discovery()
        elif choice == '2':
            return await system.run_enhanced_analysis()
        elif choice == '3':
            return await system.run_enhanced_queries()
        elif choice == '4':
            system.show_enhanced_status()
            return True
        else:
            print(f"❌ Invalid choice: '{choice}'. Please enter 0-4.")
            return True
    except KeyboardInterrupt:
        print("\n⏸️ Operation interrupted")
        return True
    except Exception as e:
        print(f"❌ Operation failed: {e}")
        print("💡 Check system status or restart the application")
        return True

def main():
    """Enhanced main entry point with architecture integration"""
    print("🚀 ENHANCED SEMANTIC DATABASE RAG SYSTEM")
    print("SQL Server Architecture + Cross-Industry + Safety Validation")
    print("Simple, Readable, Maintainable - Enhanced with Architecture Requirements")
    print("=" * 75)
    
    try:
        system = SemanticRAG()
    except Exception:
        print("\n❌ Enhanced system initialization failed")
        print("💡 Check configuration and dependencies")
        return
    
    # Enhanced main loop
    while True:
        show_enhanced_menu()
        
        try:
            choice = input("Enter choice (0-4): ").strip()
            
            if choice == '0':
                print("👋 Thanks for using Enhanced Semantic Database RAG!")
                print("🏗️ Architecture achievements:")
                print("   • ✅ SQL Server sys.* metadata integration")
                print("   • ✅ RDL business context parsing")
                print("   • ✅ Cross-industry entity taxonomy")
                print("   • ✅ Fixed SQL templates with safety validation")
                print("   • ✅ sqlglot SQL injection prevention")
                break
            
            asyncio.run(handle_enhanced_choice(system, choice))
            
        except KeyboardInterrupt:
            print("\n⏸️ Interrupted")
            break
        except Exception as e:
            print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()