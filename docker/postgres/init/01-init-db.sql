-- PostgreSQL initialization script for Trading Platform
-- This script is run only once when the database is first created

-- Create extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pgcrypto";

-- Create additional schemas if needed
-- CREATE SCHEMA IF NOT EXISTS analytics;
-- CREATE SCHEMA IF NOT EXISTS reporting;

-- Set timezone
SET timezone = 'UTC';

-- Create custom types (if needed)
-- CREATE TYPE trade_side AS ENUM ('BUY', 'SELL');
-- CREATE TYPE trade_status AS ENUM ('PENDING', 'FILLED', 'CANCELED', 'REJECTED');

-- Performance optimizations
-- Set shared_preload_libraries = 'pg_stat_statements';

-- Create indexes that will be useful for the trading platform
-- These will be created by SQLAlchemy migrations, but we can prepare the database

-- Grant permissions to trading user (already done by Docker, but for reference)
-- GRANT ALL PRIVILEGES ON DATABASE trading_db TO trading_user;
-- GRANT ALL ON SCHEMA public TO trading_user;

-- Set default statistics target for better query planning
ALTER DATABASE trading_db SET default_statistics_target = 100;

-- Enable additional logging for monitoring
ALTER DATABASE trading_db SET log_statement = 'mod';
ALTER DATABASE trading_db SET log_min_duration_statement = 1000;

-- Create a function to update updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Comment for documentation
COMMENT ON DATABASE trading_db IS 'Automated Trading Platform Database';

-- Log successful initialization
DO $$
BEGIN
    RAISE NOTICE 'Trading Platform database initialized successfully at %', now();
END $$;