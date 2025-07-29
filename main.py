#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ENHANCED SEMANTIC DATABASE RAG SYSTEM - Main Entry Point
Modular architecture with improved maintainability
"""

import asyncio
import os
from pathlib import Path

# Import modules
from db.discovery import DatabaseDiscovery
from semantic.analysis import SemanticAnalyzer
from interactive.query_interface import QueryInterface
from shared.config import Config
from shared.models import SystemStatus

def load_environment():
    """Load environment variables from .env file"""
    env_file = Path('.env')
    if env_file.exists():
        with open(env_file, 'r', encoding='utf-8') as f:
            for line in f:
                if '=' in line and not line.startswith('#'):
                    key, value = line.strip().split('=', 1)
                    os.environ[key] = value.strip('"\'')

class SemanticRAGSystem:
    """Main system orchestrator"""
    
    def __init__(self):
        self.config = Config()
        self.discovery = DatabaseDiscovery(self.config)
        self.analyzer = SemanticAnalyzer(self.config)
        self.query_interface = QueryInterface(self.config)
        self.status = SystemStatus()
    
    async def run_step1_discovery(self, limit: int = None):
        """Run Step 1: Database Discovery"""
        print("🔍 Step 1: Enhanced Database Discovery")
        print("=" * 50)
        
        success = await self.discovery.discover_database(limit)
        if success:
            self.status.discovery_completed = True
            self.status.tables_discovered = len(self.discovery.get_tables())
            print(f"✅ Discovery completed! Found {self.status.tables_discovered} objects")
        else:
            print("❌ Discovery failed")
        
        return success
    
    async def run_step2_analysis(self):
        """Run Step 2: Semantic Analysis"""
        print("\n🧠 Step 2: Enhanced Semantic Analysis")
        print("=" * 50)
        
        if not self.status.discovery_completed:
            # Try to load from cache
            if not self.discovery.load_from_cache():
                print("❌ No discovery data found. Run Step 1 first.")
                return False
        
        # Pass discovery results to analyzer
        tables = self.discovery.get_tables()
        success = await self.analyzer.analyze_semantics(tables)
        
        if success:
            self.status.analysis_completed = True
            self.status.relationships_found = len(self.analyzer.get_relationships())
            domain = self.analyzer.get_domain()
            print(f"✅ Analysis completed! Domain: {domain.domain_type if domain else 'Unknown'}")
        else:
            print("❌ Analysis failed")
        
        return success
    
    async def run_step3_queries(self):
        """Run Step 3: Interactive Queries"""
        print("\n💬 Step 3: Enhanced Interactive Queries")
        print("=" * 50)
        
        if not self.status.analysis_completed:
            # Try to load from cache
            if not (self.discovery.load_from_cache() and self.analyzer.load_from_cache()):
                print("❌ System not ready. Run Steps 1 and 2 first.")
                return False
        
        # Pass semantic analysis results to query interface
        tables = self.discovery.get_tables()
        domain = self.analyzer.get_domain()
        relationships = self.analyzer.get_relationships()
        
        await self.query_interface.start_interactive_session(tables, domain, relationships)
    
    def show_status(self):
        """Show comprehensive system status"""
        print("\n📊 ENHANCED SYSTEM STATUS")
        print("=" * 50)
        
        # Try to load data from cache if not in memory
        if not self.status.discovery_completed:
            self.discovery.load_from_cache()
        if not self.status.analysis_completed:
            self.analyzer.load_from_cache()
        
        # Show discovery status
        tables = self.discovery.get_tables()
        if tables:
            table_count = sum(1 for t in tables if t.object_type == 'BASE TABLE')
            view_count = sum(1 for t in tables if t.object_type == 'VIEW')
            print(f"📋 Database Objects: {len(tables)}")
            print(f"   Tables: {table_count}")
            print(f"   Views: {view_count}")
            
            # Data quality
            objects_with_data = sum(1 for t in tables if t.sample_data)
            print(f"   ✅ Objects with sample data: {objects_with_data}")
        
        # Show semantic analysis status
        domain = self.analyzer.get_domain()
        relationships = self.analyzer.get_relationships()
        if domain:
            print(f"\n🏢 Business Domain:")
            print(f"   Type: {domain.domain_type}")
            print(f"   Industry: {domain.industry}")
            print(f"   Confidence: {domain.confidence:.2f}")
        
        if relationships:
            print(f"   🔗 Relationships: {len(relationships)}")
        
        # Cache status
        self._show_cache_status()
    
    def _show_cache_status(self):
        """Show cache file status"""
        cache_files = [
            'data/database_structure.json',
            'data/semantic_analysis.json'
        ]
        
        print(f"\n💾 Cache Status:")
        for cache_file in cache_files:
            cache_path = Path(cache_file)
            if cache_path.exists():
                from datetime import datetime
                try:
                    modified = datetime.fromtimestamp(cache_path.stat().st_mtime)
                    age = datetime.now() - modified
                    print(f"   ✅ {cache_file}: {age.days} days old")
                except:
                    print(f"   ✅ {cache_file}: Available")
            else:
                print(f"   ⚠️  {cache_file}: Not found")

def main():
    """Enhanced main application entry point"""
    print("🧠 ENHANCED SEMANTIC DATABASE RAG SYSTEM")
    print("Modular Architecture - Simple, Readable, Maintainable")
    print("=" * 70)
    
    # Load environment
    load_environment()
    
    # Initialize system
    try:
        system = SemanticRAGSystem()
    except Exception as e:
        print(f"❌ Failed to initialize system: {e}")
        print("💡 Check your .env configuration:")
        print("   - AZURE_OPENAI_API_KEY")
        print("   - DATABASE_CONNECTION_STRING")
        print("   - AZURE_ENDPOINT")
        print("   - DEPLOYMENT_NAME")
        return
    
    print("✅ System initialized successfully")
    
    while True:
        print("\n" + "="*60)
        print("MENU OPTIONS:")
        print("1. 🔍 Database Discovery (with FAST option)")
        print("2. 🧠 Semantic Analysis (enhanced classification)")
        print("3. 💬 Interactive Queries (improved SQL generation)")
        print("4. 🚀 Full Demo (run all steps)")
        print("5. 📊 Show System Status")
        print("6. 🧪 Performance Information")
        print("0. Exit")
        print("="*60)
        
        try:
            choice = input("Enter your choice (0-6): ").strip()
            
            if choice == '0':
                print("👋 Thanks for using the Enhanced Semantic Database RAG System!")
                break
            
            elif choice == '1':
                limit_input = input("Limit objects (Enter for all, or number): ").strip()
                limit = int(limit_input) if limit_input else None
                asyncio.run(system.run_step1_discovery(limit))
            
            elif choice == '2':
                asyncio.run(system.run_step2_analysis())
            
            elif choice == '3':
                asyncio.run(system.run_step3_queries())
            
            elif choice == '4':
                async def full_demo():
                    print("🚀 Running full enhanced demonstration...")
                    
                    success1 = await system.run_step1_discovery()
                    if not success1:
                        return
                    
                    success2 = await system.run_step2_analysis()
                    if not success2:
                        return
                    
                    system.show_status()
                    
                    print("\n✅ System ready! Starting interactive mode...")
                    await system.run_step3_queries()
                
                asyncio.run(full_demo())
            
            elif choice == '5':
                system.show_status()
            
            elif choice == '6':
                print("\n🧪 PERFORMANCE FEATURES")
                print("=" * 40)
                print("✅ FAST Option Integration:")
                print("   • OPTION (FAST n) for quick view sampling")
                print("   • Enhanced view row estimation")
                print("   • Optimized query execution")
                print("\n✅ Unicode Support:")
                print("   • Proper Greek text handling")
                print("   • UTF-8 encoding throughout")
                print("\n✅ Caching System:")
                print("   • JSON cache for discovery results")
                print("   • Semantic analysis persistence")
                print("   • Configurable cache expiration")
            
            else:
                print(f"❌ Invalid choice: '{choice}'. Please enter 0-6.")
        
        except KeyboardInterrupt:
            print("\n⏸️ Interrupted by user")
            break
        except ValueError:
            print("❌ Please enter a valid number")
        except Exception as e:
            print(f"❌ Unexpected error: {e}")

if __name__ == "__main__":
    main()