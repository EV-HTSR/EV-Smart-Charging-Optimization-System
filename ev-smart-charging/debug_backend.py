from fastapi import FastAPI
import uvicorn
from sqlalchemy import create_engine, text
import pandas as pd
from fastapi.responses import JSONResponse

app = FastAPI()

# Create database engine
engine = create_engine('sqlite:///ev_charging.db')

@app.get("/debug-test")
async def debug_test():
    """Simple debug endpoint to test database connection"""
    try:
        print("🧪 Running debug test...")
        
        # Test 1: Basic connection
        with engine.connect() as conn:
            result = conn.execute(text("SELECT 1 as test_value"))
            test_value = result.scalar()
            print(f"✅ Database connection test passed: {test_value}")
        
        # Test 2: Check if stations table exists
        tables = pd.read_sql_query("SELECT name FROM sqlite_master WHERE type='table'", engine)
        print("Tables in database:", tables['name'].tolist() if not tables.empty else "No tables")
        
        # Test 3: Check stations table
        if not tables.empty and 'stations' in tables['name'].values:
            stations = pd.read_sql_query("SELECT COUNT(*) as count FROM stations", engine)
            print(f"✅ Stations table: {stations.iloc[0]['count']} records")
        else:
            print("❌ Stations table does not exist")
            return JSONResponse(
                status_code=500,
                content={"error": "Stations table does not exist. Run: python data_pipeline.py"}
            )
        
        # Test 4: Check sessions table
        if not tables.empty and 'charging_sessions' in tables['name'].values:
            sessions = pd.read_sql_query("SELECT COUNT(*) as count FROM charging_sessions", engine)
            print(f"✅ Sessions table: {sessions.iloc[0]['count']} records")
        else:
            print("❌ Charging_sessions table does not exist")
            return JSONResponse(
                status_code=500,
                content={"error": "Charging_sessions table does not exist. Run: python data_pipeline.py"}
            )
        
        return {"status": "success", "message": "All tests passed"}
        
    except Exception as e:
        error_msg = f"Debug test failed: {str(e)}"
        print(f"❌ {error_msg}")
        return JSONResponse(
            status_code=500,
            content={"error": error_msg}
        )

if __name__ == "__main__":
    print("🚀 Starting debug server on port 8001...")
    uvicorn.run(app, host="0.0.0.0", port=8001, log_level="info")