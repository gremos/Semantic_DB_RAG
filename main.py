#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Semantic Database RAG System - Simple Main Entry Point
Follows DRY, SOLID, YAGNI principles
"""

import asyncio
import os
from pathlib import Path

class SemanticRAG:
    """Simple system orchestrator following single responsibility principle"""
    
    def __init__(self):
        self.load_env()
        self.init_components()
        print("✅ Semantic Database RAG System initialized")
    
    def load_env(self):
        """Load environment variables from .env file"""
        env_file = Path('.env')
        if env_file.exists():
            with open(env_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if '=' in line and not line.startswith('#'):
                        key, value = line.strip().split('=', 1)
                        os.environ[key] = value.strip('"\'')
    
    def init_components(self):
        """Initialize system components"""
        from shared.config import Config
        from db.discovery import DatabaseDiscovery
        from semantic.analysis import SemanticAnalyzer
        from interactive.query_interface import QueryInterface
        
        self.config = Config()
        self.discovery = DatabaseDiscovery(self.config)
        self.analyzer = SemanticAnalyzer(self.config)
        self.query_interface = QueryInterface(self.config)
    
    async def run_discovery(self):
        """Option 1: Database Discovery"""
        print("🔍 DATABASE DISCOVERY")
        print("=" * 50)
        
        success = await self.discovery.discover_database()
        
        if success:
            tables = self.discovery.get_tables()
            relationships = self.discovery.get_relationships()
            print(f"✅ Discovery completed!")
            print(f"   📊 Tables: {len(tables)}")
            print(f"   🔗 Relationships: {len(relationships)}")
            return True
        else:
            print("❌ Discovery failed")
            return False
    
    async def run_analysis(self):
        """Option 2: Semantic Analysis"""
        print("\n🧠 SEMANTIC ANALYSIS")
        print("=" * 50)
        
        # Load tables from discovery or cache
        tables = self.discovery.get_tables()
        if not tables:
            if self.discovery.load_from_cache():
                tables = self.discovery.get_tables()
                print(f"   📊 Loaded {len(tables)} tables from cache")
            else:
                print("❌ No tables found. Run discovery first.")
                return False
        
        print(f"🧠 Analyzing {len(tables)} tables...")
        
        success = await self.analyzer.analyze_tables(tables)
        
        if success:
            print("✅ Semantic analysis completed!")
            domain = self.analyzer.get_domain()
            if domain:
                print(f"   🏢 Domain: {domain.domain_type}")
            return True
        else:
            print("❌ Analysis failed")
            return False
    
    async def run_queries(self):
        """Option 3: Interactive Queries (4-Stage Pipeline)"""
        print("\n💬 INTERACTIVE QUERIES")
        print("=" * 50)
        
        # Load analyzed data
        tables = self.analyzer.get_tables()
        domain = self.analyzer.get_domain()
        relationships = self.analyzer.get_relationships()
        
        # Try to load from cache if not available
        if not tables:
            if self.analyzer.load_from_cache():
                tables = self.analyzer.get_tables()
                domain = self.analyzer.get_domain()
                relationships = self.analyzer.get_relationships()
            elif self.discovery.load_from_cache():
                tables = self.discovery.get_tables()
        
        if not tables:
            print("❌ No tables available. Please run:")
            print("   1. Database Discovery")
            print("   2. Semantic Analysis")
            return False
        
        print(f"🚀 Starting 4-stage pipeline with {len(tables)} tables")
        if domain:
            print(f"   🏢 Domain: {domain.domain_type}")
        
        await self.query_interface.start_interactive_session(tables, domain, relationships)
        return True
    
    def show_status(self):
        """Option 4: System Status"""
        print("\n📊 SYSTEM STATUS")
        print("=" * 40)
        
        # Check discovery status
        discovery_tables = self.discovery.get_tables()
        if not discovery_tables:
            self.discovery.load_from_cache()
            discovery_tables = self.discovery.get_tables()
        
        # Check analysis status
        analyzer_tables = self.analyzer.get_tables()
        if not analyzer_tables:
            self.analyzer.load_from_cache()
            analyzer_tables = self.analyzer.get_tables()
        
        domain = self.analyzer.get_domain()
        relationships = self.analyzer.get_relationships()
        
        # Display status
        if discovery_tables:
            print(f"📋 Discovery: {len(discovery_tables)} objects")
        else:
            print("📋 Discovery: Not completed")
        
        if analyzer_tables:
            print(f"🧠 Analysis: {len(analyzer_tables)} tables analyzed")
        else:
            print("🧠 Analysis: Not completed")
        
        if relationships:
            print(f"🔗 Relationships: {len(relationships)}")
        
        if domain:
            print(f"🏢 Domain: {domain.domain_type}")
            if domain.capabilities:
                capabilities = [k for k, v in domain.capabilities.items() if v]
                if capabilities:
                    print(f"💼 Capabilities: {', '.join(capabilities[:3])}")

def main():
    """Clean main entry point"""
    print("🚀 SEMANTIC DATABASE RAG SYSTEM")
    print("Simple, Readable, and Maintainable")
    print("=" * 60)
    
    # Initialize system
    try:
        system = SemanticRAG()
    except Exception as e:
        print(f"❌ Failed to initialize: {e}")
        print("💡 Check your .env file:")
        print("   - AZURE_OPENAI_API_KEY")
        print("   - DATABASE_CONNECTION_STRING")
        return
    
    # Main menu loop
    while True:
        print("\n" + "="*60)
        print("MENU:")
        print("1. 🔍 Database Discovery")
        print("2. 🧠 Semantic Analysis")  
        print("3. 💬 Interactive Queries (4-Stage Pipeline)")
        print("4. 📊 System Status")
        print("0. Exit")
        print("="*60)
        
        try:
            choice = input("Enter your choice (0-4): ").strip()
            
            if choice == '0':
                print("👋 Thanks for using the Semantic Database RAG System!")
                break
            elif choice == '1':
                asyncio.run(system.run_discovery())
            elif choice == '2':
                asyncio.run(system.run_analysis())
            elif choice == '3':
                asyncio.run(system.run_queries())
            elif choice == '4':
                system.show_status()
            else:
                print(f"❌ Invalid choice: '{choice}'. Please enter 0-4.")
        
        except KeyboardInterrupt:
            print("\n⏸️ Interrupted")
            break
        except Exception as e:
            print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()