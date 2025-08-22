#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Semantic Database RAG System - Simple, Readable, Maintainable
Following README: DRY, SOLID, YAGNI principles
Clean function names: SemanticRAG (not SimplifiedSemanticRAG)
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

class SystemValidator:
    """System validation - Single responsibility"""
    
    @staticmethod
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
            print("\n💡 Check your .env file settings")
            return False
        
        print("✅ Configuration validated")
        return True

class ComponentManager:
    """Component management - Single responsibility"""
    
    def __init__(self, config):
        self.config = config
        self.components = {}
    
    def load_components(self) -> bool:
        """Load all system components"""
        try:
            from db.discovery import DatabaseDiscovery
            from semantic.analysis import SemanticAnalyzer
            from interactive.query_interface import QueryInterface
            
            self.components = {
                'discovery': DatabaseDiscovery(self.config),
                'analyzer': SemanticAnalyzer(self.config), 
                'query_interface': QueryInterface(self.config)
            }
            
            print("✅ Components loaded")
            return True
            
        except Exception as e:
            print(f"❌ Component loading failed: {e}")
            return False
    
    def get(self, name: str):
        """Get component by name"""
        return self.components.get(name)

class WorkflowManager:
    """Workflow operations - Single responsibility"""
    
    def __init__(self, components: ComponentManager):
        self.components = components
    
    async def run_discovery(self) -> bool:
        """Run database discovery"""
        print("\n🔍 DATABASE DISCOVERY")
        print("=" * 50)
        
        try:
            discovery = self.components.get('discovery')
            success = await discovery.discover_database()
            
            if success:
                stats = discovery.get_discovery_stats()
                print(f"✅ Discovery completed!")
                print(f"   📊 Tables: {stats['tables']}")
                print(f"   👁️ Views: {stats['views']}")
                return True
            else:
                print("❌ Discovery failed")
                return False
                
        except Exception as e:
            print(f"❌ Discovery error: {e}")
            return False
    
    async def run_analysis(self) -> bool:
        """Run semantic analysis"""
        print("\n🧠 SEMANTIC ANALYSIS")
        print("=" * 50)
        
        try:
            discovery = self.components.get('discovery')
            analyzer = self.components.get('analyzer')
            
            # Load tables
            tables = discovery.get_tables()
            if not tables:
                if discovery.load_from_cache():
                    tables = discovery.get_tables()
                else:
                    print("❌ No tables found. Run Discovery first.")
                    return False
            
            success = await analyzer.analyze_tables(tables)
            
            if success:
                stats = analyzer.get_analysis_stats()
                print("✅ Analysis completed!")
                print(f"   📊 Total tables: {stats['total_tables']}")
                print(f"   📈 Fact tables: {stats['fact_tables']}")
                return True
            else:
                print("❌ Analysis failed")
                return False
                
        except Exception as e:
            print(f"❌ Analysis error: {e}")
            return False
    
    async def run_queries(self) -> bool:
        """Run interactive queries"""
        print("\n💬 INTERACTIVE QUERIES")
        print("=" * 50)
        
        try:
            analyzer = self.components.get('analyzer')
            discovery = self.components.get('discovery')
            query_interface = self.components.get('query_interface')
            
            # Load data
            tables = []
            domain = None
            relationships = []
            
            # Try analyzer first
            if hasattr(analyzer, 'get_tables'):
                tables = analyzer.get_tables()
                domain = analyzer.get_domain()
                relationships = analyzer.get_relationships()
            
            # Try caches if no data
            if not tables:
                print("   🔄 Loading from caches...")
                
                if analyzer.load_from_cache():
                    tables = analyzer.get_tables()
                    domain = analyzer.get_domain()
                    relationships = analyzer.get_relationships()
                elif discovery.load_from_cache():
                    tables = discovery.get_tables()
                    relationships = discovery.get_relationships()
            
            if not tables:
                print("❌ No data available. Please run:")
                print("   1. Database Discovery")
                print("   2. Semantic Analysis")
                return False
            
            print(f"🚀 Starting pipeline:")
            print(f"   📊 Tables: {len(tables)}")
            print(f"   🔗 Relationships: {len(relationships)}")
            
            await query_interface.start_session(tables, domain, relationships)
            return True
            
        except Exception as e:
            print(f"❌ Query interface error: {e}")
            return False
    
    def show_status(self) -> None:
        """Show system status"""
        print("\n📊 SYSTEM STATUS")
        print("=" * 30)
        
        try:
            discovery = self.components.get('discovery')
            analyzer = self.components.get('analyzer')
            
            # Check discovery
            discovery_tables = discovery.get_tables()
            if not discovery_tables:
                discovery.load_from_cache()
                discovery_tables = discovery.get_tables()
            
            # Check analysis
            analyzer_tables = []
            if hasattr(analyzer, 'load_from_cache'):
                analyzer.load_from_cache()
                analyzer_tables = analyzer.get_tables()
            
            # Show status
            print(f"📋 Discovery: {'✅ Complete' if discovery_tables else '❌ Incomplete'}")
            if discovery_tables:
                stats = discovery.get_discovery_stats()
                print(f"   📊 {stats['total_objects']} objects")
            
            print(f"🧠 Analysis: {'✅ Complete' if analyzer_tables else '❌ Incomplete'}")
            if analyzer_tables:
                print(f"   📊 {len(analyzer_tables)} tables")
            
            print(f"💬 Pipeline: {'✅ Ready' if (analyzer_tables or discovery_tables) else '❌ Not ready'}")
            
        except Exception as e:
            print(f"⚠️ Status check error: {e}")

class SemanticRAG:
    """Main system orchestrator - Clean interface"""
    
    def __init__(self):
        load_env()
        self.config = None
        self.components = None
        self.workflow = None
        self._initialize()
    
    def _initialize(self):
        """Initialize system"""
        try:
            # Check dependencies
            if not SystemValidator.check_dependencies():
                raise RuntimeError("Missing dependencies")
            
            # Load config
            from shared.config import Config
            self.config = Config()
            
            # Validate config
            if not SystemValidator.check_config(self.config):
                raise RuntimeError("Configuration validation failed")
            
            # Load components
            self.components = ComponentManager(self.config)
            if not self.components.load_components():
                raise RuntimeError("Component loading failed")
            
            # Initialize workflow
            self.workflow = WorkflowManager(self.components)
            
            print("✅ SemanticRAG System initialized")
            print("   Simple, Readable, Maintainable")
            
        except Exception as e:
            print(f"❌ Initialization failed: {e}")
            print("\n💡 Setup checklist:")
            print("   1. Copy env_example.txt to .env")
            print("   2. Set AZURE_OPENAI_API_KEY")
            print("   3. Set DATABASE_CONNECTION_STRING")
            print("   4. Set AZURE_ENDPOINT")
            print("   5. Install: pip install -r requirements.txt")
            raise
    
    async def run_discovery(self) -> bool:
        """Run discovery"""
        return await self.workflow.run_discovery()
    
    async def run_analysis(self) -> bool:
        """Run analysis"""
        return await self.workflow.run_analysis()
    
    async def run_queries(self) -> bool:
        """Run queries"""
        return await self.workflow.run_queries()
    
    def show_status(self) -> None:
        """Show status"""
        self.workflow.show_status()

def show_menu() -> None:
    """Display main menu"""
    print("\n" + "="*50)
    print("SEMANTIC DATABASE RAG SYSTEM")
    print("Simple, Readable, Maintainable")
    print("="*50)
    print("1. 🔍 Database Discovery")
    print("2. 🧠 Semantic Analysis")  
    print("3. 💬 Interactive Queries")
    print("4. 📊 System Status")
    print("0. Exit")
    print("="*50)

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
    """Main entry point - Clean and simple"""
    print("🚀 SEMANTIC DATABASE RAG SYSTEM")
    print("Simple, Readable, and Maintainable")
    print("Following README instructions")
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
                print("👋 Thanks for using Semantic Database RAG!")
                break
            
            success = asyncio.run(handle_choice(system, choice))
            
        except KeyboardInterrupt:
            print("\n⏸️ Interrupted")
            break
        except Exception as e:
            print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()