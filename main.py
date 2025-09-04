#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Semantic Database RAG System - Enhanced & Fixed
Simple function names: SemanticRAG (not SimplifiedSemanticRAG)
Following DRY, SOLID, YAGNI principles - Pure LLM approach with fixes
"""

import asyncio
import os
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
    try:
        import pyodbc
        from langchain_openai import AzureChatOpenAI
        print("✅ Dependencies available")
        return True
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("💡 Install: pip install pyodbc langchain-openai sqlglot")
        return False

class SemanticRAG:
    """Main system orchestrator - Enhanced & Fixed"""
    
    def __init__(self):
        load_env()
        self._initialize()
    
    def _initialize(self):
        """Initialize system components"""
        try:
            # Check dependencies
            if not check_dependencies():
                raise RuntimeError("Missing dependencies")
            
            # Load configuration
            from shared.config import Config
            self.config = Config()
            
            # Validate configuration
            health = self.config.get_health_check()
            if not health['llm_configured'] or not health['database_configured']:
                raise RuntimeError("Configuration incomplete")
            
            # Initialize components
            from db.discovery import DatabaseDiscovery
            from semantic.analysis import SemanticAnalyzer  # Enhanced with views
            from interactive.query_interface import QueryInterface  # Fixed SQL
            
            self.discovery = DatabaseDiscovery(self.config)
            self.analyzer = SemanticAnalyzer(self.config)
            self.query_interface = QueryInterface(self.config)
            
            print("✅ Enhanced SemanticRAG System initialized")
            print("   📊 Database discovery with first 3 + last 3 sampling")
            print("   🧠 Enhanced LLM analysis (tables + views + columns)")
            print("   💬 Fixed 3-stage query pipeline")
            
        except Exception as e:
            print(f"❌ Initialization failed: {e}")
            print("\n💡 Setup checklist:")
            print("   1. Copy env_example.txt to .env")
            print("   2. Set AZURE_OPENAI_API_KEY")
            print("   3. Set DATABASE_CONNECTION_STRING")
            print("   4. Set AZURE_ENDPOINT")
            raise
    
    async def run_discovery(self) -> bool:
        """Run enhanced database discovery"""
        print("\n🔍 ENHANCED DATABASE DISCOVERY")
        print("=" * 35)
        
        try:
            success = await self.discovery.discover_database()
            
            if success:
                stats = self.discovery.get_discovery_stats()
                print(f"✅ Discovery completed!")
                print(f"   📊 Tables: {stats['tables']}")
                print(f"   👁️ Views: {stats['views']}")
                print(f"   ⚙️ Stored Procedures: {stats['stored_procedures']}")
                print(f"   📝 Sampling: First 3 + Last 3 rows")
                print(f"   🔍 View definitions: {'✅ Included' if stats['views'] > 0 else '❌ None'}")
                return True
            else:
                print("❌ Discovery failed")
                return False
                
        except Exception as e:
            print(f"❌ Discovery error: {e}")
            return False
    
    async def run_analysis(self) -> bool:
        """Run enhanced LLM semantic analysis"""
        print("\n🧠 ENHANCED SEMANTIC ANALYSIS")
        print("=" * 35)
        
        try:
            # Get tables from discovery
            tables = self.discovery.get_tables()
            if not tables:
                if self.discovery.load_from_cache():
                    tables = self.discovery.get_tables()
                else:
                    print("❌ No tables found. Run Discovery first.")
                    return False
            
            # Run enhanced LLM analysis with view information
            success = await self.analyzer.analyze_tables(tables, self.discovery)
            
            if success:
                stats = self.analyzer.get_analysis_stats()
                print("✅ Enhanced LLM analysis completed!")
                print(f"   📊 Total objects: {stats['total_objects']}")
                print(f"   📋 Tables: {stats['total_tables']}")
                print(f"   👁️ Views: {stats['total_views']}")
                print(f"   👥 Customer objects: {stats['customer_tables']}")
                print(f"   💳 Payment objects: {stats['payment_tables']}")
                print(f"   📈 Fact tables: {stats['fact_tables']}")
                print(f"   🧠 Method: Enhanced LLM with deep column analysis")
                return True
            else:
                print("❌ Analysis failed")
                return False
                
        except Exception as e:
            print(f"❌ Analysis error: {e}")
            return False
    
    async def run_queries(self) -> bool:
        """Run enhanced query interface"""
        print("\n💬 ENHANCED QUERY INTERFACE")
        print("=" * 35)
        
        try:
            # Load data
            tables = []
            domain = None
            relationships = []
            
            # Try analyzer first (has enriched tables)
            if hasattr(self.analyzer, 'get_tables'):
                tables = self.analyzer.get_tables()
                domain = self.analyzer.get_domain()
                relationships = self.analyzer.get_relationships()
            
            # Fallback to caches
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
                print("   1. Database Discovery")
                print("   2. Semantic Analysis")
                return False
            
            # Show enhanced readiness
            entities = {}
            for table in tables:
                entity = table.entity_type
                entities[entity] = entities.get(entity, 0) + 1
            
            print(f"🚀 Starting enhanced pipeline:")
            print(f"   📊 Total objects: {len(tables)}")
            
            # Show top entities
            sorted_entities = sorted(entities.items(), key=lambda x: x[1], reverse=True)[:5]
            for entity, count in sorted_entities:
                print(f"   🎯 {entity}: {count} objects")
            
            print(f"   🔄 Pipeline: Enhanced Intent → Smart Selection → Fixed SQL")
            
            await self.query_interface.start_session(tables, domain, relationships)
            return True
            
        except Exception as e:
            print(f"❌ Query interface error: {e}")
            return False
    
    def show_status(self) -> None:
        """Show enhanced system status"""
        print("\n📊 ENHANCED SYSTEM STATUS")
        print("=" * 30)
        
        try:
            # Check discovery
            discovery_tables = self.discovery.get_tables()
            if not discovery_tables:
                self.discovery.load_from_cache()
                discovery_tables = self.discovery.get_tables()
            
            # Check analysis
            analyzer_tables = []
            if hasattr(self.analyzer, 'load_from_cache'):
                self.analyzer.load_from_cache()
                analyzer_tables = self.analyzer.get_tables()
            
            # Show enhanced status
            print(f"📋 Discovery: {'✅ Ready' if discovery_tables else '❌ Incomplete'}")
            if discovery_tables:
                stats = self.discovery.get_discovery_stats()
                print(f"   📊 {stats['total_objects']} objects discovered")
                print(f"   📋 Tables: {stats['tables']}")
                print(f"   👁️ Views: {stats['views']}")
                print(f"   ⚙️ Procedures: {stats['stored_procedures']}")
            
            print(f"🧠 Analysis: {'✅ Ready' if analyzer_tables else '❌ Incomplete'}")
            if analyzer_tables:
                stats = self.analyzer.get_analysis_stats()
                print(f"   📊 {stats['total_objects']} objects analyzed")
                print(f"   📋 Tables: {stats['total_tables']}")
                print(f"   👁️ Views: {stats['total_views']}")
                print(f"   👥 Customer objects: {stats.get('customer_tables', 0)}")
                print(f"   💳 Payment objects: {stats.get('payment_tables', 0)}")
                print(f"   📈 Fact tables: {stats.get('fact_tables', 0)}")
                print(f"   🧠 Method: {stats.get('analysis_method', 'enhanced_llm')}")
            
            print(f"💬 Queries: {'✅ Ready' if (analyzer_tables or discovery_tables) else '❌ Not ready'}")
            
            # Show available entities
            if analyzer_tables:
                entities = {}
                for table in analyzer_tables:
                    entity = table.entity_type
                    entities[entity] = entities.get(entity, 0) + 1
                
                print(f"\n🎯 Available Entities:")
                sorted_entities = sorted(entities.items(), key=lambda x: x[1], reverse=True)
                for entity, count in sorted_entities[:8]:
                    print(f"   • {entity}: {count} objects")
            
        except Exception as e:
            print(f"⚠️ Status check error: {e}")

def show_menu() -> None:
    """Display enhanced main menu"""
    print("\n" + "="*55)
    print("SEMANTIC DATABASE RAG SYSTEM - ENHANCED")
    print("Enhanced LLM Analysis & Fixed SQL Generation")
    print("="*55)
    print("1. 🔍 Database Discovery (Tables + Views + Procedures)")
    print("2. 🧠 Enhanced Analysis (Deep Column + View Integration)")  
    print("3. 💬 Fixed Query Interface (Smart Selection + Fixed SQL)")
    print("4. 📊 Enhanced System Status")
    print("0. Exit")
    print("="*55)
    print("💡 Key Enhancements:")
    print("   • Deep table+column analysis")
    print("   • View integration in analysis")
    print("   • Fixed SQL generation templates")
    print("   • Smart multi-factor table selection")
    print("   • Enhanced business entity detection")

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
    """Main entry point"""
    print("🚀 SEMANTIC DATABASE RAG SYSTEM - ENHANCED")
    print("Simple, Readable, Maintainable - Now with Fixes!")
    print("=" * 50)
    
    try:
        system = SemanticRAG()
    except Exception:
        print("\n❌ System initialization failed")
        return
    
    # Main loop
    while True:
        show_menu()
        
        try:
            choice = input("Enter choice (0-4): ").strip()
            
            if choice == '0':
                print("👋 Thanks for using Enhanced Semantic Database RAG!")
                print("   🧠 Enhanced LLM analysis with views")
                print("   ⚡ Fixed SQL generation")
                print("   📊 Smart table selection")
                break
            
            asyncio.run(handle_choice(system, choice))
            
        except KeyboardInterrupt:
            print("\n⏸️ Interrupted")
            break
        except Exception as e:
            print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()