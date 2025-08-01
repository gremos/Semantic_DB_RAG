#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fixed Semantic Database RAG System - Main Entry Point
Fixes table loading issue between discovery and semantic analysis
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
    """Fixed system orchestrator with proper table management"""
    
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
        """Step 2: Semantic Analysis with proper table loading"""
        print("\n🧠 Step 2: Semantic Analysis")
        print("=" * 50)
        
        # FIXED: Ensure we have tables first
        tables = self.discovery.get_tables()
        
        if not tables:
            # Try to load from discovery cache
            print("   📁 No tables in memory, loading from discovery cache...")
            if self.discovery.load_from_cache():
                tables = self.discovery.get_tables()
                print(f"   📊 Loaded {len(tables)} tables from discovery cache")
            else:
                print("❌ No tables found. Run discovery first.")
                return False
        
        print(f"🧠 Analyzing {len(tables)} tables for business entities and relationships")
        
        success = await self.analyzer.analyze_tables(tables)
        
        if success:
            print("✅ Semantic analysis completed!")
            
            # FIXED: Verify tables are properly stored
            analyzed_tables = self.analyzer.get_tables()
            print(f"   📊 Analysis result: {len(analyzed_tables)} tables classified")
            
            if len(analyzed_tables) == 0:
                print("   ⚠️ Warning: No tables in analyzer, keeping original tables")
                # Keep original tables if analysis didn't store them properly
                self.analyzer.tables = tables
            
            return True
        else:
            print("❌ Semantic analysis failed")
            return False
    
    async def run_interactive_queries(self):
        """Step 3: Interactive Queries with improved table loading"""
        print("\n💬 Step 3: Interactive Queries")
        print("=" * 50)
        
        # FIXED: Better table loading logic
        tables = None
        
        # Try to get tables from analyzer first
        if hasattr(self.analyzer, 'tables') and self.analyzer.get_tables():
            tables = self.analyzer.get_tables()
            print(f"   📊 Using {len(tables)} tables from semantic analysis")
        
        # Fallback to discovery tables
        elif self.discovery.get_tables():
            tables = self.discovery.get_tables()
            print(f"   📊 Using {len(tables)} tables from discovery")
        
        # Try to load from caches
        else:
            print("   📁 No tables in memory, trying to load from cache...")
            
            # Try semantic cache first
            if self.analyzer.load_from_cache():
                tables = self.analyzer.get_tables()
                if tables:
                    print(f"   📊 Loaded {len(tables)} tables from semantic cache")
            
            # Try discovery cache as fallback
            if not tables and self.discovery.load_from_cache():
                tables = self.discovery.get_tables()
                if tables:
                    print(f"   📊 Loaded {len(tables)} tables from discovery cache")
        
        if not tables:
            print("❌ No tables available. Please run:")
            print("   1. Database Discovery (option 1)")
            print("   2. Semantic Analysis (option 2)")
            return False
        
        # Get analysis results
        domain = self.analyzer.get_domain()
        relationships = self.analyzer.get_relationships()
        
        print(f"🚀 Starting interactive session with {len(tables)} tables")
        if domain:
            print(f"   🏢 Domain: {domain.domain_type}")
        if relationships:
            print(f"   🔗 Relationships: {len(relationships)}")
        
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
        """Show current system status with better diagnostics"""
        print("\n📊 SYSTEM STATUS")
        print("=" * 40)
        
        # Discovery status
        discovery_tables = self.discovery.get_tables()
        if discovery_tables:
            table_count = sum(1 for t in discovery_tables if t.object_type in ['BASE TABLE', 'TABLE'])
            view_count = sum(1 for t in discovery_tables if t.object_type == 'VIEW')
            print(f"📋 Discovery: {len(discovery_tables)} objects ({table_count} tables, {view_count} views)")
        else:
            print("📋 Discovery: Not completed")
        
        # Semantic analysis status
        analyzer_tables = self.analyzer.get_tables()
        domain = self.analyzer.get_domain()
        relationships = self.analyzer.get_relationships()
        
        if analyzer_tables:
            classified_count = sum(1 for t in analyzer_tables if t.entity_type != 'Unknown')
            print(f"🧠 Semantic Analysis: {len(analyzer_tables)} tables, {classified_count} classified")
        else:
            print("🧠 Semantic Analysis: No tables loaded")
        
        if relationships:
            print(f"🔗 Relationships: {len(relationships)} discovered")
        else:
            print("🔗 Relationships: None found")
        
        if domain:
            print(f"🏢 Domain: {domain.domain_type} (confidence: {domain.confidence:.2f})")
            
            # Show entity distribution if we have tables
            tables_for_analysis = analyzer_tables or discovery_tables
            if tables_for_analysis:
                entity_counts = {}
                for table in tables_for_analysis:
                    if hasattr(table, 'entity_type') and table.entity_type != 'Unknown':
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