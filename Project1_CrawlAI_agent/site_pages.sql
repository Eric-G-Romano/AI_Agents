-- Enable the pgvector extension
CREATE EXTENSION IF NOT EXISTS VECTOR;

-- Create the documentation chunks table
CREATE TABLE SITE_PAGES (
    ID BIGSERIAL PRIMARY KEY,
    URL VARCHAR NOT NULL,
    CHUNK_NUMBER INTEGER NOT NULL,
    TITLE VARCHAR NOT NULL,
    SUMMARY VARCHAR NOT NULL,
    CONTENT VARCHAR NOT NULL, -- Added content column
    METADATA JSONB NOT NULL DEFAULT '{}'::JSONB, -- Added metadata column
    EMBEDDING VECTOR(1536), -- OpenAI embeddings are 1536 dimensions
    CREATED_AT TIMESTAMP WITH TIME ZONE DEFAULT TIMEZONE('utc'::TEXT, NOW()) NOT NULL,
    -- Add a unique contraint to prevent duplicate chunks for the same URL
    UNIQUE(URL, CHUNK_NUMBER)
);

-- Create an index for better vector similarity search performance
CREATE INDEX ON SITE_PAGES USING IVFFLAT(EMBEDDING VECTOR_COSINE_OPS);

-- Create an index on metadata for faster filering
CREATE INDEX IDX_SITE_PAGES_METADATA ON SITE_PAGES USING GIN(METADATA);

-- Create a function to search for documentation chunks
CREATE FUNCTION MATCH_SITE_PAGES(
    QUERY_EMBEDDING VECTOR(1536),
    MATCH_COUNT INT DEFAULT 10,
    FILTER JSONB DEFAULT '{}'::JSONB
) RETURNS TABLE (
    ID BIGINT,
    URL VARCHAR,
    CHUNK_NUMBER INTEGER,
    TITLE VARCHAR,
    SUMMARY VARCHAR,
    CONTENT VARCHAR,
    METADATA JSONB,
    SIMILARITY FLOAT
) 
LANGUAGE plpgsql
AS $$
#variable_conflict use_column
BEGIN
    RETURN QUERY
    SELECT
        ID,
        URL,
        CHUNK_NUMBER,
        TITLE,
        SUMMARY,
        CONTENT,
        METADATA,
        1-(SITE_PAGES.EMBEDDING <=> QUERY_EMBEDDING) AS SIMILARITY
    FROM
        SITE_PAGES
    WHERE
        METADATA @> FILTER
    ORDER BY
        SITE_PAGES.EMBEDDING <=> QUERY_EMBEDDING
    LIMIT
        MATCH_COUNT;
END;
$$;

-- Everything below is for Supabase security

-- Enable RLS on the table
ALTER TABLE SITE_PAGES ENABLE ROW LEVEL SECURITY;

-- Create a policy that allows anyone to read
CREATE POLICY "Allow public read access"
    ON SITE_PAGES
    FOR SELECT
    TO PUBLIC
    USING (true);