#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Enhanced Semantic Database RAG System - Simple, Readable, Maintainable
Following README: DRY, SOLID, YAGNI principles
Clean function names: SemanticRAG (not SimplifiedSemanticRAG)
Enhanced with better entity resolution and customer/payment focus
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
    """Enhanced system orchestrator with better entity resolution"""
    
    def __init__(self):
        load_env()
        self._initialize()
    
    def _initialize(self):
        """Initialize enhanced system"""
        try:
            # Check dependencies
            if not check_dependencies():
                raise RuntimeError("Missing dependencies")
            
            # Load config
            from shared.config import Config
            self.config = Config()
            
            # Validate config
            health = self.config.get_health_check()
            if not health['llm_configured'] or not health['database_configured']:
                raise RuntimeError("Configuration incomplete")
            
            # Load enhanced components
            from db.discovery import DatabaseDiscovery
            from semantic.analysis import EnhancedSemanticAnalyzer
            from interactive.query_interface import QueryInterface
            
            self.discovery = DatabaseDiscovery(self.config)
            self.analyzer = EnhancedSemanticAnalyzer(self.config)  # Use enhanced version
            self.query_interface = QueryInterface(self.config)
            
            print("✅ Enhanced SemanticRAG System initialized")
            print("   Enhanced with LLM entity resolution")
            print("   Better customer/payment analysis")
            
        except Exception as e:
            print(f"❌ Enhanced initialization failed: {e}")
            print("\n💡 Setup checklist:")
            print("   1. Copy env_example.txt to .env")
            print("   2. Set AZURE_OPENAI_API_KEY")
            print("   3. Set DATABASE_CONNECTION_STRING")
            print("   4. Set AZURE_ENDPOINT")
            print("   5. Install: pip install -r requirements.txt")
            raise
    
    async def run_discovery(self) -> bool:
        """Run enhanced discovery"""
        print("\n🔍 ENHANCED DATABASE DISCOVERY")
        print("=" * 50)
        
        try:
            success = await self.discovery.discover_database()
            
            if success:
                stats = self.discovery.get_discovery_stats()
                print(f"✅ Enhanced discovery completed!")
                print(f"   📊 Tables: {stats['tables']}")
                print(f"   👁️ Views: {stats['views']}")
                print(f"   📝 Sampling: First 3 + Last 3 rows")
                return True
            else:
                print("❌ Enhanced discovery failed")
                return False
                
        except Exception as e:
            print(f"❌ Enhanced discovery error: {e}")
            return False
    
    async def run_analysis(self) -> bool:
        """Run enhanced analysis"""
        print("\n🧠 ENHANCED SEMANTIC ANALYSIS")
        print("=" * 50)
        
        try:
            # Load tables
            tables = self.discovery.get_tables()
            if not tables:
                if self.discovery.load_from_cache():
                    tables = self.discovery.get_tables()
                else:
                    print("❌ No tables found. Run Discovery first.")
                    return False
            
            success = await self.analyzer.analyze_tables(tables)
            
            if success:
                stats = self.analyzer.get_analysis_stats()
                print("✅ Enhanced analysis completed!")
                print(f"   📊 Total tables: {stats['total_tables']}")
                print(f"   👥 Customer tables: {stats['customer_tables']}")
                print(f"   💳 Payment tables: {stats['payment_tables']}")
                print(f"   📈 Fact tables: {stats['fact_tables']}")
                return True
            else:
                print("❌ Enhanced analysis failed")
                return False
                
        except Exception as e:
            print(f"❌ Enhanced analysis error: {e}")
            return False
    
    async def run_queries(self) -> bool:
        """Run enhanced queries with LLM entity resolution"""
        print("\n💬 ENHANCED INTERACTIVE QUERIES")
        print("=" * 50)
        
        try:
            # Load enhanced data
            tables = []
            domain = None
            relationships = []
            
            # Try enhanced analyzer first
            if hasattr(self.analyzer, 'get_tables'):
                tables = self.analyzer.get_tables()
                domain = self.analyzer.get_domain()
                relationships = self.analyzer.get_relationships()
            
            # Try caches if no data
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
                print("   1. Enhanced Database Discovery")
                print("   2. Enhanced Semantic Analysis")
                return False
            
            # Show enhanced readiness
            customer_tables = len([t for t in tables if hasattr(t, 'is_customer_table') and t.is_customer_table()])
            payment_tables = len([t for t in tables if hasattr(t, 'is_payment_table') and t.is_payment_table()])
            
            print(f"🚀 Starting enhanced pipeline:")
            print(f"   📊 Total tables: {len(tables)}")
            print(f"   👥 Customer tables: {customer_tables}")
            print(f"   💳 Payment tables: {payment_tables}")
            print(f"   🔗 Relationships: {len(relationships)}")
            print(f"   🧠 LLM entity resolution: Enabled")
            
            await self.query_interface.start_session(tables, domain, relationships)
            return True
            
        except Exception as e:
            print(f"❌ Enhanced query interface error: {e}")
            return False
    
    def show_status(self) -> None:
        """Show enhanced status"""
        print("\n📊 ENHANCED SYSTEM STATUS")
        print("=" * 30)
        
        try:
            # Check discovery
            discovery_tables = self.discovery.get_tables()
            if not discovery_tables:
                self.discovery.load_from_cache()
                discovery_tables = self.discovery.get_tables()
            
            # Check enhanced analysis
            analyzer_tables = []
            if hasattr(self.analyzer, 'load_from_cache'):
                self.analyzer.load_from_cache()
                analyzer_tables = self.analyzer.get_tables()
            
            # Show enhanced status
            print(f"📋 Discovery: {'✅ Complete' if discovery_tables else '❌ Incomplete'}")
            if discovery_tables:
                stats = self.discovery.get_discovery_stats()
                print(f"   📊 {stats['total_objects']} objects discovered")
                print(f"   📝 Enhanced sampling: First 3 + Last 3")
            
            print(f"🧠 Analysis: {'✅ Complete' if analyzer_tables else '❌ Incomplete'}")
            if analyzer_tables:
                try:
                    stats = self.analyzer.get_analysis_stats()
                    print(f"   📊 {len(analyzer_tables)} tables analyzed")
                    print(f"   👥 {stats.get('customer_tables', 0)} customer tables")
                    print(f"   💳 {stats.get('payment_tables', 0)} payment tables")
                except Exception as e:
                    print(f"   📊 {len(analyzer_tables)} tables analyzed")
            
            print(f"💬 Pipeline: {'✅ Ready' if (analyzer_tables or discovery_tables) else '❌ Not ready'}")
            
            if analyzer_tables or discovery_tables:
                print(f"🧠 Enhanced Features:")
                print(f"   • LLM-powered entity resolution")
                print(f"   • Customer/payment focus")
                print(f"   • Intent-driven SQL generation")
            
        except Exception as e:
            print(f"⚠️ Enhanced status check error: {e}")


def show_enhanced_menu() -> None:
    """Display enhanced main menu"""
    print("\n" + "="*60)
    print("ENHANCED SEMANTIC DATABASE RAG SYSTEM")
    print("LLM-Powered Entity Resolution & Customer Analytics")
    print("="*60)
    print("1. 🔍 Enhanced Database Discovery")
    print("2. 🧠 Enhanced Semantic Analysis")  
    print("3. 💬 Enhanced Interactive Queries")
    print("4. 📊 Enhanced System Status")
    print("0. Exit")
    print("="*60)
    print("💡 Enhanced Features:")
    print("   • Better customer/payment table recognition")
    print("   • LLM-powered intent analysis")
    print("   • Schema-aware entity resolution")
    print("   • Simplified SQL generation")


async def handle_enhanced_choice(system: SemanticRAG, choice: str) -> bool:
    """Handle enhanced menu choice"""
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
        print(f"❌ Enhanced operation failed: {e}")
        return True


def main():
    """Enhanced main entry point"""
    print("🚀 ENHANCED SEMANTIC DATABASE RAG SYSTEM")
    print("LLM-Powered Entity Resolution & Customer Analytics")
    print("Following README with enhanced capabilities")
    print("=" * 60)
    
    try:
        system = SemanticRAG()
    except Exception:
        print("\n❌ Enhanced system initialization failed")
        return
    
    # Enhanced main loop
    while True:
        show_enhanced_menu()
        
        try:
            choice = input("Enter choice (0-4): ").strip()
            
            if choice == '0':
                print("👋 Thanks for using Enhanced Semantic Database RAG!")
                print("   🧠 LLM entity resolution enabled")
                print("   👥 Customer analytics ready")
                print("   💳 Payment analysis supported")
                break
            
            success = asyncio.run(handle_enhanced_choice(system, choice))
            
        except KeyboardInterrupt:
            print("\n⏸️ Interrupted")
            break
        except Exception as e:
            print(f"❌ Enhanced error: {e}")


if __name__ == "__main__":
    main()