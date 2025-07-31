#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simplified Semantic Database RAG System - Main Entry Point
Simple, readable, and maintainable implementation
"""

import asyncio
import os
from pathlib import Path

# Import simplified modules
from db.discovery import DatabaseDiscovery
from semantic.analysis import SimpleSemanticAnalyzer
from interactive.query_interface import SimpleQueryInterface
from shared.config import Config

def load_environment():
    """Load environment variables from .env file"""
    env_file = Path('.env')
    if env_file.exists():
        with open(env_file, 'r', encoding='utf-8') as f:
            for line in f:
                if '=' in line and not line.startswith('#'):
                    key, value = line.strip().split('=', 1)
                    os.environ[key] = value.strip('"\'')

class SimpleSemanticRAGSystem:
    """Simplified system orchestrator focusing on core functionality"""
    
    def __init__(self):
        self.config = Config()
        self.discovery = DatabaseDiscovery(self.config)
        self.analyzer = SimpleSemanticAnalyzer(self.config)
        self.query_interface = SimpleQueryInterface(self.config)
        
        print("✅ Simple Semantic RAG System initialized")
    
    async def run_database_discovery(self, limit: int = None):
        """Step 1: Database Discovery"""
        print("🔍 Step 1: Database Discovery")
        print("=" * 50)
        
        if limit:
            print(f"📊 Analyzing top {limit} priority objects")
        else:
            print("📊 Analyzing ALL database objects")
        
        success = await self.discovery.discover_database(limit)
        
        if success:
            tables = self.discovery.get_tables()
            table_count = sum(1 for t in tables if t.object_type in ['BASE TABLE', 'TABLE'])
            view_count = sum(1 for t in tables if t.object_type == 'VIEW')
            
            print(f"✅ Discovery completed!")
            print(f"   📊 Found {len(tables)} objects ({table_count} tables, {view_count} views)")
            
            return True
        else:
            print("❌ Discovery failed")
            return False
    
    async def run_semantic_analysis(self):
        """Step 2: Semantic Analysis"""
        print("\n🧠 Step 2: Semantic Analysis")
        print("=" * 50)
        
        # Get tables from discovery
        tables = self.discovery.get_tables()
        if not tables:
            # Try to load from cache
            if not self.discovery.load_from_cache():
                print("❌ No tables found. Run discovery first.")
                return False
            tables = self.discovery.get_tables()
        
        print(f"🧠 Analyzing {len(tables)} tables for business entities and relationships")
        
        success = await self.analyzer.analyze_tables(tables)
        
        if success:
            # Update tables with semantic information
            analyzed_tables = self.analyzer.get_tables()
            self.discovery.tables = analyzed_tables  # Update discovery tables
            
            print("✅ Semantic analysis completed!")
            return True
        else:
            print("❌ Semantic analysis failed")
            return False
    
    async def run_interactive_queries(self):
        """Step 3: Interactive Queries"""
        print("\n💬 Step 3: Interactive Queries")
        print("=" * 50)
        
        # Get analysis results
        tables = self.analyzer.get_tables() or self.discovery.get_tables()
        domain = self.analyzer.get_domain()
        relationships = self.analyzer.get_relationships()
        
        if not tables:
            print("❌ No tables available. Run discovery and analysis first.")
            return False
        
        print(f"🚀 Starting interactive session with {len(tables)} tables")
        
        # Start interactive session
        await self.query_interface.start_interactive_session(
            tables, domain, relationships
        )
        
        return True
    
    async def run_full_demo(self):
        """Run complete demonstration"""
        print("🚀 Full Demo - Complete Analysis Pipeline")
        print("=" * 60)
        
        confirm = input("Run full analysis (discovery + semantic + queries)? (y/N): ").strip().lower()
        if confirm != 'y':
            print("Demo cancelled")
            return
        
        print("\n🚀 Starting full analysis pipeline...")
        
        # Step 1: Discovery
        success1 = await self.run_database_discovery()
        if not success1:
            print("❌ Pipeline stopped at discovery")
            return
        
        # Step 2: Semantic Analysis
        success2 = await self.run_semantic_analysis()
        if not success2:
            print("❌ Pipeline stopped at semantic analysis")
            return
        
        # Show status
        self.show_system_status()
        
        print("\n✅ Analysis pipeline completed!")
        print("🎯 System ready for intelligent queries")
        
        # Step 3: Interactive Queries
        await self.run_interactive_queries()
    
    def show_system_status(self):
        """Show current system status"""
        print("\n📊 SYSTEM STATUS")
        print("=" * 40)
        
        # Discovery status
        tables = self.discovery.get_tables()
        if tables:
            table_count = sum(1 for t in tables if t.object_type in ['BASE TABLE', 'TABLE'])
            view_count = sum(1 for t in tables if t.object_type == 'VIEW')
            print(f"📋 Discovery: {len(tables)} objects ({table_count} tables, {view_count} views)")
        else:
            print("📋 Discovery: Not completed")
        
        # Semantic analysis status
        domain = self.analyzer.get_domain()
        relationships = self.analyzer.get_relationships()
        
        if domain:
            classified_count = sum(1 for t in tables if t.entity_type != 'Unknown')
            print(f"🧠 Semantic Analysis: {classified_count} entities classified")
            print(f"🔗 Relationships: {len(relationships)} discovered")
            print(f"🏢 Domain: {domain.domain_type} (confidence: {domain.confidence:.2f})")
            
            # Show entity distribution
            entity_counts = {}
            for table in tables:
                if table.entity_type != 'Unknown':
                    entity_counts[table.entity_type] = entity_counts.get(table.entity_type, 0) + 1
            
            if entity_counts:
                print(f"📊 Business Entities:")
                for entity_type, count in sorted(entity_counts.items()):
                    print(f"   • {entity_type}: {count}")
            
            # Show capabilities
            if domain.capabilities:
                enabled_caps = [cap for cap, enabled in domain.capabilities.items() if enabled]
                if enabled_caps:
                    print(f"🎯 Query Capabilities:")
                    for cap in enabled_caps:
                        print(f"   ✅ {cap.replace('_', ' ').title()}")
        else:
            print("🧠 Semantic Analysis: Not completed")
        
        # Cache status
        cache_files = [
            ('data/database_structure.json', 'Discovery'),
            ('data/semantic_analysis.json', 'Semantic Analysis')
        ]
        
        print(f"💾 Cache Status:")
        for cache_file, description in cache_files:
            cache_path = Path(cache_file)
            if cache_path.exists():
                size_mb = cache_path.stat().st_size / (1024 * 1024)
                print(f"   ✅ {description}: {size_mb:.1f}MB")
            else:
                print(f"   ❌ {description}: Not cached")

def main():
    """Main application entry point"""
    print("🚀 SIMPLIFIED SEMANTIC DATABASE RAG SYSTEM")
    print("Simple, Readable, and Maintainable")
    print("=" * 60)
    
    # Load environment
    load_environment()
    
    # Initialize system
    try:
        system = SimpleSemanticRAGSystem()
    except Exception as e:
        print(f"❌ Failed to initialize system: {e}")
        print("💡 Check your .env configuration:")
        print("   - AZURE_OPENAI_API_KEY")
        print("   - DATABASE_CONNECTION_STRING")
        print("   - AZURE_ENDPOINT")
        print("   - DEPLOYMENT_NAME")
        return
    
    while True:
        print("\n" + "="*60)
        print("SIMPLIFIED SEMANTIC DATABASE RAG SYSTEM:")
        print("1. 🔍 Database Discovery - Find and analyze database objects")
        print("2. 🎯 Limited Discovery - Analyze specific number of objects")
        print("3. 🧠 Semantic Analysis - Classify entities and find relationships")
        print("4. 💬 Interactive Queries - Natural language querying")
        print("5. 🚀 Full Demo - Complete analysis pipeline")
        print("6. 📊 System Status - Show current analysis state")
        print("0. Exit")
        print("="*60)
        
        try:
            choice = input("Enter your choice (0-6): ").strip()
            
            if choice == '0':
                print("👋 Thanks for using the Simplified Semantic Database RAG System!")
                break
            
            elif choice == '1':
                print("\n🔍 FULL DISCOVERY MODE")
                print("This will analyze ALL database objects")
                confirm = input("Continue with full discovery? (y/N): ").strip().lower()
                if confirm == 'y':
                    asyncio.run(system.run_database_discovery())
            
            elif choice == '2':
                try:
                    limit = int(input("Enter number of objects to analyze: ").strip())
                    if limit > 0:
                        asyncio.run(system.run_database_discovery(limit))
                    else:
                        print("❌ Please enter a positive number")
                except ValueError:
                    print("❌ Please enter a valid number")
            
            elif choice == '3':
                asyncio.run(system.run_semantic_analysis())
            
            elif choice == '4':
                asyncio.run(system.run_interactive_queries())
            
            elif choice == '5':
                asyncio.run(system.run_full_demo())
            
            elif choice == '6':
                system.show_system_status()
            
            else:
                print(f"❌ Invalid choice: '{choice}'. Please enter 0-6.")
        
        except KeyboardInterrupt:
            print("\n⏸️ Interrupted by user")
            break
        except Exception as e:
            print(f"❌ Unexpected error: {e}")

if __name__ == "__main__":
    main()