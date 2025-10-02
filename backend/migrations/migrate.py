"""
Database migration script for the trading platform.
Creates and updates database schema.
"""

import asyncio
import sys
from pathlib import Path
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy import text
import logging

# Add the parent directory to the path so we can import backend modules
sys.path.append(str(Path(__file__).parent.parent.parent))

from backend.models.database import Base, DatabaseManager
from backend.config.settings import get_settings
from backend.utils.logger import TradingLogger

# Initialize logger
logger = TradingLogger()

class DatabaseMigrator:
    """Handles database migrations and schema updates"""
    
    def __init__(self, database_url: str):
        self.database_url = database_url
        self.engine = create_async_engine(database_url, echo=False)
    
    async def create_tables(self):
        """Create all database tables"""
        try:
            logger.info("Creating database tables...")
            
            async with self.engine.begin() as conn:
                # Create all tables
                await conn.run_sync(Base.metadata.create_all)
            
            logger.info("Database tables created successfully")
            
        except Exception as e:
            logger.error(f"Error creating database tables: {str(e)}")
            raise
    
    async def check_connection(self):
        """Check database connection"""
        try:
            async with self.engine.connect() as conn:
                result = await conn.execute(text("SELECT 1"))
                await result.fetchone()
            logger.info("Database connection successful")
            return True
        except Exception as e:
            logger.error(f"Database connection failed: {str(e)}")
            return False
    
    async def get_schema_version(self):
        """Get current schema version"""
        try:
            async with self.engine.connect() as conn:
                # Check if migration_history table exists
                result = await conn.execute(text("""
                    SELECT EXISTS (
                        SELECT FROM information_schema.tables 
                        WHERE table_schema = 'public' 
                        AND table_name = 'migration_history'
                    );
                """))
                
                table_exists = await result.fetchone()
                
                if not table_exists[0]:
                    # Create migration_history table
                    await conn.execute(text("""
                        CREATE TABLE migration_history (
                            id SERIAL PRIMARY KEY,
                            version VARCHAR(50) NOT NULL UNIQUE,
                            applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                            description TEXT
                        );
                    """))
                    await conn.commit()
                    logger.info("Created migration_history table")
                    return "0.0.0"
                
                # Get latest version
                result = await conn.execute(text("""
                    SELECT version FROM migration_history 
                    ORDER BY applied_at DESC LIMIT 1;
                """))
                
                row = await result.fetchone()
                return row[0] if row else "0.0.0"
                
        except Exception as e:
            logger.error(f"Error getting schema version: {str(e)}")
            return "0.0.0"
    
    async def record_migration(self, version: str, description: str):
        """Record a completed migration"""
        try:
            async with self.engine.begin() as conn:
                await conn.execute(text("""
                    INSERT INTO migration_history (version, description)
                    VALUES (:version, :description);
                """), {"version": version, "description": description})
            
            logger.info(f"Recorded migration: {version} - {description}")
            
        except Exception as e:
            logger.error(f"Error recording migration: {str(e)}")
            raise
    
    async def run_migration_001(self):
        """Migration 001: Initial schema setup"""
        version = "1.0.0"
        description = "Initial schema setup with all core tables"
        
        try:
            logger.info(f"Running migration {version}: {description}")
            
            # Create all tables using SQLAlchemy models
            await self.create_tables()
            
            # Create additional indexes for performance
            async with self.engine.begin() as conn:
                # Indexes for trades table
                await conn.execute(text("""
                    CREATE INDEX IF NOT EXISTS idx_trades_symbol ON trades(symbol);
                    CREATE INDEX IF NOT EXISTS idx_trades_timestamp ON trades(timestamp);
                    CREATE INDEX IF NOT EXISTS idx_trades_status ON trades(status);
                    CREATE INDEX IF NOT EXISTS idx_trades_user_id ON trades(user_id);
                """))
                
                # Indexes for signals table
                await conn.execute(text("""
                    CREATE INDEX IF NOT EXISTS idx_signals_symbol ON signals(symbol);
                    CREATE INDEX IF NOT EXISTS idx_signals_timestamp ON signals(timestamp);
                    CREATE INDEX IF NOT EXISTS idx_signals_action ON signals(action);
                """))
                
                # Indexes for portfolio table
                await conn.execute(text("""
                    CREATE INDEX IF NOT EXISTS idx_portfolio_symbol ON portfolio(symbol);
                    CREATE INDEX IF NOT EXISTS idx_portfolio_user_id ON portfolio(user_id);
                """))
                
                # Indexes for performance_metrics table
                await conn.execute(text("""
                    CREATE INDEX IF NOT EXISTS idx_performance_timestamp ON performance_metrics(timestamp);
                    CREATE INDEX IF NOT EXISTS idx_performance_timeframe ON performance_metrics(timeframe);
                """))
            
            # Record the migration
            await self.record_migration(version, description)
            
            logger.info(f"Migration {version} completed successfully")
            
        except Exception as e:
            logger.error(f"Error in migration {version}: {str(e)}")
            raise
    
    async def run_migration_002(self):
        """Migration 002: Add additional columns and constraints"""
        version = "1.0.1"
        description = "Add additional columns for enhanced functionality"
        
        try:
            logger.info(f"Running migration {version}: {description}")
            
            async with self.engine.begin() as conn:
                # Add new columns to trades table
                await conn.execute(text("""
                    ALTER TABLE trades 
                    ADD COLUMN IF NOT EXISTS commission DECIMAL(10, 6) DEFAULT 0,
                    ADD COLUMN IF NOT EXISTS slippage DECIMAL(10, 6) DEFAULT 0,
                    ADD COLUMN IF NOT EXISTS order_type VARCHAR(20) DEFAULT 'MARKET';
                """))
                
                # Add new columns to signals table
                await conn.execute(text("""
                    ALTER TABLE signals 
                    ADD COLUMN IF NOT EXISTS model_predictions JSONB,
                    ADD COLUMN IF NOT EXISTS technical_indicators JSONB;
                """))
                
                # Add constraints
                await conn.execute(text("""
                    ALTER TABLE trades 
                    ADD CONSTRAINT IF NOT EXISTS chk_trades_quantity_positive 
                    CHECK (quantity > 0);
                    
                    ALTER TABLE trades 
                    ADD CONSTRAINT IF NOT EXISTS chk_trades_price_positive 
                    CHECK (price > 0);
                """))
            
            # Record the migration
            await self.record_migration(version, description)
            
            logger.info(f"Migration {version} completed successfully")
            
        except Exception as e:
            logger.error(f"Error in migration {version}: {str(e)}")
            raise
    
    async def run_all_migrations(self):
        """Run all pending migrations"""
        try:
            # Check database connection
            if not await self.check_connection():
                raise Exception("Cannot connect to database")
            
            current_version = await self.get_schema_version()
            logger.info(f"Current schema version: {current_version}")
            
            # Define migration sequence
            migrations = [
                ("1.0.0", self.run_migration_001),
                ("1.0.1", self.run_migration_002),
            ]
            
            # Run pending migrations
            for version, migration_func in migrations:
                if self._version_compare(current_version, version) < 0:
                    await migration_func()
                    current_version = version
                else:
                    logger.info(f"Skipping migration {version} (already applied)")
            
            logger.info("All migrations completed successfully")
            
        except Exception as e:
            logger.error(f"Migration failed: {str(e)}")
            raise
    
    def _version_compare(self, version1: str, version2: str) -> int:
        """Compare two version strings. Returns -1 if v1 < v2, 0 if equal, 1 if v1 > v2"""
        v1_parts = [int(x) for x in version1.split('.')]
        v2_parts = [int(x) for x in version2.split('.')]
        
        # Pad shorter version with zeros
        max_len = max(len(v1_parts), len(v2_parts))
        v1_parts.extend([0] * (max_len - len(v1_parts)))
        v2_parts.extend([0] * (max_len - len(v2_parts)))
        
        for v1, v2 in zip(v1_parts, v2_parts):
            if v1 < v2:
                return -1
            elif v1 > v2:
                return 1
        
        return 0
    
    async def close(self):
        """Close database connection"""
        await self.engine.dispose()

async def main():
    """Main migration function"""
    try:
        # Get settings
        settings = get_settings()
        
        # Create migrator
        migrator = DatabaseMigrator(settings.database_url)
        
        # Run migrations
        await migrator.run_all_migrations()
        
        # Close connection
        await migrator.close()
        
        logger.info("Database migration completed successfully")
        
    except Exception as e:
        logger.error(f"Database migration failed: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())