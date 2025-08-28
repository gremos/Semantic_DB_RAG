#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Semantic Database RAG System - Simplified & Maintainable
Simple function names: SemanticRAG (not SimplifiedSemanticRAG)
Following DRY, SOLID, YAGNI principles - Pure LLM approach
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
    """Main system orchestrator - Simple and Clean"""
    
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
            from semantic.analysis import SemanticAnalyzer  # Uses pure LLM approach
            from interactive.query_interface import QueryInterface  # Simplified pipeline
            
            self.discovery = DatabaseDiscovery(self.config)
            self.analyzer = SemanticAnalyzer(self.config)
            self.query_interface = QueryInterface(self.config)
            
            print("✅ SemanticRAG System initialized")
            print("   📊 Database discovery with first 3 + last 3 sampling")
            print("   🧠 Pure LLM semantic analysis") 
            print("   💬 Simplified 3-stage query pipeline")
            
        except Exception as e:
            print(f"❌ Initialization failed: {e}")
            print("\n💡 Setup checklist:")
            print("   1. Copy env_example.txt to .env")
            print("   2. Set AZURE_OPENAI_API_KEY")
            print("   3. Set DATABASE_CONNECTION_STRING")
            print("   4. Set AZURE_ENDPOINT")
            raise
    
    async def run_discovery(self) -> bool:
        """Run database discovery"""
        print("\n🔍 DATABASE DISCOVERY")
        print("=" * 30)
        
        try:
            success = await self.discovery.discover_database()
            
            if success:
                stats = self.discovery.get_discovery_stats()
                print(f"✅ Discovery completed!")
                print(f"   📊 Tables: {stats['tables']}")
                print(f"   👁️ Views: {stats['views']}")
                print(f"   📝 Sampling: First 3 + Last 3 rows")
                return True
            else:
                print("❌ Discovery failed")
                return False
                
        except Exception as e:
            print(f"❌ Discovery error: {e}")
            return False
    
    async def run_analysis(self) -> bool:
        """Run pure LLM semantic analysis"""
        print("\n🧠 SEMANTIC ANALYSIS")
        print("=" * 30)
        
        try:
            # Get tables from discovery
            tables = self.discovery.get_tables()
            if not tables:
                if self.discovery.load_from_cache():
                    tables = self.discovery.get_tables()
                else:
                    print("❌ No tables found. Run Discovery first.")
                    return False
            
            # Run pure LLM analysis
            success = await self.analyzer.analyze_tables(tables)
            
            if success:
                stats = self.analyzer.get_analysis_stats()
                print("✅ Pure LLM analysis completed!")
                print(f"   📊 Tables analyzed: {stats['total_tables']}")
                print(f"   👥 Customer tables: {stats['customer_tables']}")
                print(f"   💳 Payment tables: {stats['payment_tables']}")
                print(f"   🧠 Method: Pure LLM")
                return True
            else:
                print("❌ Analysis failed")
                return False
                
        except Exception as e:
            print(f"❌ Analysis error: {e}")
            return False
    
    async def run_queries(self) -> bool:
        """Run simplified query interface"""
        print("\n💬 QUERY INTERFACE")
        print("=" * 30)
        
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
            
            # Show readiness
            customer_tables = len([t for t in tables if getattr(t, 'entity_type', '') == 'Customer'])
            payment_tables = len([t for t in tables if getattr(t, 'entity_type', '') == 'Payment'])
            
            print(f"🚀 Starting simplified pipeline:")
            print(f"   📊 Tables: {len(tables)}")
            print(f"   👥 Customer tables: {customer_tables}")
            print(f"   💳 Payment tables: {payment_tables}")
            print(f"   🧠 Pipeline: Intent → Tables → SQL")
            
            await self.query_interface.start_session(tables, domain, relationships)
            return True
            
        except Exception as e:
            print(f"❌ Query interface error: {e}")
            return False
    
    def show_status(self) -> None:
        """Show system status"""
        print("\n📊 SYSTEM STATUS")
        print("=" * 20)
        
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
            
            # Show status
            print(f"📋 Discovery: {'✅ Ready' if discovery_tables else '❌ Incomplete'}")
            if discovery_tables:
                stats = self.discovery.get_discovery_stats()
                print(f"   📊 {stats['total_objects']} objects discovered")
            
            print(f"🧠 Analysis: {'✅ Ready' if analyzer_tables else '❌ Incomplete'}")
            if analyzer_tables:
                stats = self.analyzer.get_analysis_stats()
                print(f"   📊 {len(analyzer_tables)} tables analyzed")
                print(f"   👥 {stats.get('customer_tables', 0)} customer tables")
                print(f"   💳 {stats.get('payment_tables', 0)} payment tables")
                print(f"   🧠 Method: {stats.get('analysis_method', 'unknown')}")
            
            print(f"💬 Queries: {'✅ Ready' if (analyzer_tables or discovery_tables) else '❌ Not ready'}")
            
        except Exception as e:
            print(f"⚠️ Status check error: {e}")

def show_menu() -> None:
    """Display main menu"""
    print("\n" + "="*50)
    print("SEMANTIC DATABASE RAG SYSTEM")
    print("Pure LLM Analysis & Simplified Pipeline")
    print("="*50)
    print("1. 🔍 Database Discovery")
    print("2. 🧠 Semantic Analysis (Pure LLM)")
    print("3. 💬 Query Interface (Simplified)")
    print("4. 📊 System Status")
    print("0. Exit")
    print("="*50)
    print("💡 Features:")
    print("   • First 3 + Last 3 sampling")
    print("   • Pure LLM table analysis")
    print("   • 3-stage query pipeline")

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
    print("🚀 SEMANTIC DATABASE RAG SYSTEM")
    print("Simple, Readable, Maintainable")
    print("=" * 40)
    
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
                print("👋 Thanks for using Semantic Database RAG!")
                print("   🧠 Pure LLM analysis")
                print("   🔄 Simplified pipeline")
                break
            
            asyncio.run(handle_choice(system, choice))
            
        except KeyboardInterrupt:
            print("\n⏸️ Interrupted")
            break
        except Exception as e:
            print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()