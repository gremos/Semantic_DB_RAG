#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simple Semantic Database RAG System
Following README: Simple, Readable, Maintainable
DRY, SOLID, YAGNI principles
"""

import asyncio
import os
from pathlib import Path

def load_env():
    """Load .env file - Simple utility function"""
    env_file = Path('.env')
    if env_file.exists():
        with open(env_file, 'r', encoding='utf-8') as f:
            for line in f:
                if '=' in line and not line.startswith('#'):
                    key, value = line.strip().split('=', 1)
                    os.environ[key] = value.strip('"\'')

class SemanticRAG:
    """Simple system orchestrator - Single responsibility"""
    
    def __init__(self):
        load_env()
        self.init_components()
    
    def init_components(self):
        """Initialize components with error handling"""
        try:
            from shared.config import Config
            from db.discovery import DatabaseDiscovery
            from semantic.analysis import SemanticAnalyzer
            from interactive.query_interface import QueryInterface
            
            self.config = Config()
            self.discovery = DatabaseDiscovery(self.config)
            self.analyzer = SemanticAnalyzer(self.config)
            self.query_interface = QueryInterface(self.config)
            
            print("✅ Semantic RAG System initialized")
            
        except Exception as e:
            print(f"❌ Initialization failed: {e}")
            raise
    
    async def run_discovery(self):
        """Option 1: Database Discovery"""
        print("\n🔍 DATABASE DISCOVERY")
        print("=" * 50)
        
        try:
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
                
        except Exception as e:
            print(f"❌ Discovery error: {e}")
            return False
    
    async def run_analysis(self):
        """Option 2: Semantic Analysis"""
        print("\n🧠 SEMANTIC ANALYSIS")
        print("=" * 50)
        
        try:
            # Load tables from discovery
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
                
        except Exception as e:
            print(f"❌ Analysis error: {e}")
            return False
    
    async def run_queries(self):
        """Option 3: Interactive Queries (4-Stage Pipeline)"""
        print("\n💬 INTERACTIVE QUERIES - 4-STAGE PIPELINE")
        print("=" * 50)
        
        try:
            # Load data for pipeline
            tables = self.analyzer.get_tables()
            domain = self.analyzer.get_domain()
            relationships = self.analyzer.get_relationships()
            
            # Try cache if not loaded
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
            
            await self.query_interface.start_session(tables, domain, relationships)
            return True
            
        except Exception as e:
            print(f"❌ Query interface error: {e}")
            return False
    
    def show_status(self):
        """Option 4: System Status"""
        print("\n📊 SYSTEM STATUS")
        print("=" * 40)
        
        try:
            # Check discovery
            discovery_tables = self.discovery.get_tables()
            if not discovery_tables:
                self.discovery.load_from_cache()
                discovery_tables = self.discovery.get_tables()
            
            # Check analysis
            analyzer_tables = self.analyzer.get_tables()
            if not analyzer_tables:
                self.analyzer.load_from_cache()
                analyzer_tables = self.analyzer.get_tables()
            
            domain = self.analyzer.get_domain()
            relationships = self.analyzer.get_relationships()
            
            # Display status
            print(f"📋 Discovery: {len(discovery_tables)} objects" if discovery_tables else "📋 Discovery: Not completed")
            print(f"🧠 Analysis: {len(analyzer_tables)} tables" if analyzer_tables else "🧠 Analysis: Not completed")
            
            if relationships:
                print(f"🔗 Relationships: {len(relationships)}")
            
            if domain:
                print(f"🏢 Domain: {domain.domain_type}")
                capabilities = [k for k, v in domain.capabilities.items() if v]
                if capabilities:
                    print(f"💼 Capabilities: {', '.join(capabilities[:3])}")
                    
        except Exception as e:
            print(f"⚠️ Status check error: {e}")

def main():
    """Simple main entry point"""
    print("🚀 SEMANTIC DATABASE RAG SYSTEM")
    print("Simple, Readable, and Maintainable")
    print("=" * 60)
    
    try:
        system = SemanticRAG()
    except Exception as e:
        print(f"❌ Failed to initialize: {e}")
        print("💡 Check your .env file:")
        print("   - AZURE_OPENAI_API_KEY")
        print("   - DATABASE_CONNECTION_STRING")
        return
    
    # Simple menu loop
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
            choice = input("Enter choice (0-4): ").strip()
            
            if choice == '0':
                print("👋 Thanks for using Semantic Database RAG!")
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