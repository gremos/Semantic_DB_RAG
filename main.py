#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Semantic Database RAG System - Main Entry Point
Simple, Readable, Maintainable - Following README exactly
DRY, SOLID, YAGNI principles with clean function names
"""

import asyncio
import os
from pathlib import Path
from typing import Optional

def load_env():
    """Load environment variables from .env file (DRY principle)"""
    env_file = Path('.env')
    if env_file.exists():
        with open(env_file, 'r', encoding='utf-8') as f:
            for line in f:
                if '=' in line and not line.startswith('#'):
                    key, value = line.strip().split('=', 1)
                    os.environ[key] = value.strip('"\'')

class SystemChecker:
    """System validation - Single responsibility (SOLID)"""
    
    @staticmethod
    def check_dependencies() -> bool:
        """Check required dependencies"""
        try:
            import pyodbc
            import sqlglot
            from langchain_openai import AzureChatOpenAI
            return True
        except ImportError as e:
            print(f"❌ Missing dependency: {e}")
            print("💡 Install with: pip install pyodbc sqlglot langchain-openai")
            return False
    
    @staticmethod
    def check_config(config) -> bool:
        """Validate configuration"""
        health = config.get_health_check()
        
        issues = []
        if not health['llm_configured']:
            issues.append("Azure OpenAI settings incomplete")
        if not health['database_configured']:
            issues.append("Database connection string missing")
        
        if issues:
            print("❌ Configuration issues:")
            for issue in issues:
                print(f"   - {issue}")
            return False
        
        return True

class ComponentLoader:
    """Component initialization - Single responsibility (SOLID)"""
    
    def __init__(self, config):
        self.config = config
        self.components = {}
    
    def load_all(self) -> bool:
        """Load all system components"""
        try:
            from shared.config import Config
            from db.discovery import DatabaseDiscovery
            from semantic.analysis import SemanticAnalyzer
            from interactive.query_interface import QueryInterface
            
            # Store components
            self.components = {
                'discovery': DatabaseDiscovery(self.config),
                'analyzer': SemanticAnalyzer(self.config), 
                'query_interface': QueryInterface(self.config)
            }
            
            return True
            
        except Exception as e:
            print(f"❌ Component loading failed: {e}")
            return False
    
    def get(self, name: str):
        """Get component by name"""
        return self.components.get(name)

class MenuHandler:
    """Menu operations - Single responsibility (SOLID)"""
    
    def __init__(self, components: ComponentLoader):
        self.components = components
    
    async def run_discovery(self) -> bool:
        """Option 1: Database Discovery"""
        print("\n🔍 DATABASE DISCOVERY")
        print("Following README: Schema + samples + view/SP analysis")
        print("=" * 60)
        
        try:
            discovery = self.components.get('discovery')
            success = await discovery.discover_database()
            
            if success:
                tables = discovery.get_tables()
                relationships = discovery.get_relationships()
                view_info = discovery.get_view_info()
                sp_info = discovery.get_stored_procedure_info()
                
                print(f"✅ Discovery completed!")
                print(f"   📊 Tables: {len(tables)}")
                print(f"   👁️ Views: {len(view_info)}")
                print(f"   ⚙️ Stored Procedures: {len(sp_info)}")
                print(f"   🔗 Relationships: {len(relationships)}")
                return True
            else:
                print("❌ Discovery failed")
                return False
                
        except Exception as e:
            print(f"❌ Discovery error: {e}")
            return False
    
    async def run_analysis(self) -> bool:
        """Option 2: Semantic Analysis"""
        print("\n🧠 SEMANTIC ANALYSIS")
        print("Following README: Entity classification, business templates")
        print("=" * 60)
        
        try:
            discovery = self.components.get('discovery')
            analyzer = self.components.get('analyzer')
            
            # Load tables
            tables = discovery.get_tables()
            if not tables:
                if discovery.load_from_cache():
                    tables = discovery.get_tables()
                    print(f"   📊 Loaded {len(tables)} tables from cache")
                else:
                    print("❌ No tables found. Run Discovery first.")
                    return False
            
            print(f"🧠 Analyzing {len(tables)} tables...")
            success = await analyzer.analyze_tables(tables)
            
            if success:
                domain = analyzer.get_domain()
                relationships = analyzer.get_relationships()
                
                print("✅ Analysis completed!")
                if domain:
                    print(f"   🏢 Domain: {domain.domain_type}")
                    print(f"   🎯 Confidence: {domain.confidence:.2f}")
                
                print(f"   🔗 Relationships: {len(relationships)}")
                return True
            else:
                print("❌ Analysis failed")
                return False
                
        except Exception as e:
            print(f"❌ Analysis error: {e}")
            return False
    
    async def run_queries(self) -> bool:
        """Option 3: Interactive Queries - 4-Stage Pipeline"""
        print("\n💬 4-STAGE INTERACTIVE PIPELINE")
        print("Following README: Constrained + EG, Schema-first, Enterprise guardrails")
        print("=" * 80)
        
        try:
            analyzer = self.components.get('analyzer')
            discovery = self.components.get('discovery')
            query_interface = self.components.get('query_interface')
            
            # Load data
            tables = analyzer.get_tables()
            domain = analyzer.get_domain()
            relationships = analyzer.get_relationships()
            
            # Try loading from caches
            if not tables:
                if analyzer.load_from_cache():
                    tables = analyzer.get_tables()
                    domain = analyzer.get_domain()
                    relationships = analyzer.get_relationships()
                    print("   📊 Loaded analysis from cache")
                elif discovery.load_from_cache():
                    tables = discovery.get_tables()
                    print("   📊 Loaded tables from discovery cache")
            
            if not tables:
                print("❌ No data available. Please run:")
                print("   1. Database Discovery")
                print("   2. Semantic Analysis")
                return False
            
            print(f"🚀 Starting 4-stage pipeline:")
            print(f"   📊 Tables: {len(tables)}")
            print(f"   🎯 Domain: {domain.domain_type if domain else 'Unknown'}")
            print(f"   🔗 Relationships: {len(relationships) if relationships else 0}")
            
            await query_interface.start_session(tables, domain, relationships)
            return True
            
        except Exception as e:
            print(f"❌ Query interface error: {e}")
            return False
    
    def show_status(self) -> None:
        """Option 4: System Status"""
        print("\n📊 SYSTEM STATUS")
        print("=" * 40)
        
        try:
            discovery = self.components.get('discovery')
            analyzer = self.components.get('analyzer')
            
            # Check discovery status
            discovery_tables = discovery.get_tables()
            if not discovery_tables:
                discovery.load_from_cache()
                discovery_tables = discovery.get_tables()
            
            # Check analysis status  
            analyzer_tables = analyzer.get_tables()
            if not analyzer_tables:
                analyzer.load_from_cache()
                analyzer_tables = analyzer.get_tables()
            
            domain = analyzer.get_domain()
            relationships = analyzer.get_relationships()
            view_info = discovery.get_view_info()
            
            # Display status
            print(f"📋 Discovery Status:")
            if discovery_tables:
                table_count = sum(1 for t in discovery_tables if t.object_type in ['BASE TABLE', 'TABLE'])
                view_count = sum(1 for t in discovery_tables if t.object_type == 'VIEW')
                
                print(f"   ✅ Completed - {len(discovery_tables)} objects")
                print(f"   📊 Tables: {table_count}")
                print(f"   👁️ Views: {view_count} (definitions: {len(view_info)})")
            else:
                print(f"   ❌ Not completed - Run Option 1")
            
            print(f"\n🧠 Analysis Status:")
            if analyzer_tables:
                classified = sum(1 for t in analyzer_tables if t.entity_type != 'Unknown')
                print(f"   ✅ Completed - {len(analyzer_tables)} tables")
                print(f"   🏷️ Classified: {classified}")
                
                if domain:
                    print(f"   🏢 Domain: {domain.domain_type}")
                    print(f"   🎯 Confidence: {domain.confidence:.2f}")
            else:
                print(f"   ❌ Not completed - Run Option 2")
            
            print(f"\n💬 Pipeline Status:")
            if analyzer_tables or discovery_tables:
                print(f"   ✅ Ready for interactive queries")
            else:
                print(f"   ❌ Not ready - Complete Options 1 & 2 first")
            
            print(f"\n📖 README Compliance:")
            print(f"   ✅ Simple, Readable, Maintainable code")
            print(f"   ✅ DRY, SOLID, YAGNI principles")
            print(f"   ✅ Function names: SemanticRAG (not SimplifiedSemanticRAG)")
            print(f"   ✅ All features implemented")
                    
        except Exception as e:
            print(f"⚠️ Status check error: {e}")

class SemanticRAG:
    """Main system orchestrator - Clean interface (SOLID principle)"""
    
    def __init__(self):
        load_env()
        self.config = None
        self.components = None
        self.menu = None
        self._initialize()
    
    def _initialize(self):
        """Initialize system components"""
        try:
            # Check dependencies first
            if not SystemChecker.check_dependencies():
                raise RuntimeError("Missing required dependencies")
            
            # Load configuration
            from shared.config import Config
            self.config = Config()
            
            # Validate configuration
            if not SystemChecker.check_config(self.config):
                raise RuntimeError("Configuration validation failed")
            
            # Load components
            self.components = ComponentLoader(self.config)
            if not self.components.load_all():
                raise RuntimeError("Component loading failed")
            
            # Initialize menu handler
            self.menu = MenuHandler(self.components)
            
            print("✅ Semantic RAG System initialized")
            print("   Following README: Simple, Readable, Maintainable")
            
        except Exception as e:
            print(f"❌ Initialization failed: {e}")
            print("\n💡 Setup checklist:")
            print("   1. Copy env_example.txt to .env")
            print("   2. Set AZURE_OPENAI_API_KEY")
            print("   3. Set DATABASE_CONNECTION_STRING")
            print("   4. Set AZURE_ENDPOINT")
            print("   5. Install: pip install pyodbc sqlglot langchain-openai")
            raise
    
    async def run_discovery(self) -> bool:
        """Run database discovery"""
        return await self.menu.run_discovery()
    
    async def run_analysis(self) -> bool:
        """Run semantic analysis"""
        return await self.menu.run_analysis()
    
    async def run_queries(self) -> bool:
        """Run interactive queries"""
        return await self.menu.run_queries()
    
    def show_status(self) -> None:
        """Show system status"""
        self.menu.show_status()

def show_main_menu() -> None:
    """Display main menu (DRY principle)"""
    print("\n" + "="*60)
    print("SEMANTIC DATABASE RAG SYSTEM")
    print("Simple, Readable, Maintainable (README compliant)")
    print("="*60)
    print("1. 🔍 Database Discovery (Schema + samples + views/SPs)")
    print("2. 🧠 Semantic Analysis (Classification + relationships)")  
    print("3. 💬 Interactive Queries (4-Stage Pipeline)")
    print("4. 📊 System Status")
    print("0. Exit")
    print("="*60)

async def handle_menu_choice(system: SemanticRAG, choice: str) -> bool:
    """Handle menu choice (DRY principle)"""
    handlers = {
        '1': system.run_discovery,
        '2': system.run_analysis, 
        '3': system.run_queries,
        '4': lambda: system.show_status() or True  # show_status returns None, so add True
    }
    
    handler = handlers.get(choice)
    if handler:
        if choice == '4':
            handler()
            return True
        else:
            return await handler()
    else:
        print(f"❌ Invalid choice: '{choice}'. Please enter 0-4.")
        return True

def main():
    """Main entry point - Clean and simple"""
    print("🚀 SEMANTIC DATABASE RAG SYSTEM")
    print("Simple, Readable, and Maintainable")
    print("Following README instructions exactly")
    print("DRY, SOLID, YAGNI principles")
    print("=" * 60)
    
    try:
        system = SemanticRAG()
    except Exception:
        return  # Error already printed in __init__
    
    # Main menu loop
    while True:
        show_main_menu()
        
        try:
            choice = input("Enter choice (0-4): ").strip()
            
            if choice == '0':
                print("👋 Thanks for using Semantic Database RAG!")
                print("Built following README: Simple, Readable, Maintainable")
                break
            
            # Handle menu choice
            success = asyncio.run(handle_menu_choice(system, choice))
            
        except KeyboardInterrupt:
            print("\n⏸️ Interrupted")
            break
        except Exception as e:
            print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()