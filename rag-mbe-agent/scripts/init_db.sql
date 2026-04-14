-- Init script executed by PostgreSQL on first container start.
-- Enables uuid-ossp for gen_random_uuid() — available in PostgreSQL 13+.

CREATE EXTENSION IF NOT EXISTS "pgcrypto";

-- Tables are created by the application on startup via SQLAlchemy.
-- This script only ensures required extensions are available.

GRANT ALL PRIVILEGES ON DATABASE mbe_agent TO mbe_user;
