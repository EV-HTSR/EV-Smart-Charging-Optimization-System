import traceback
from database import engine, Session, Station, ChargingSession

def check_database():
    # Create a session instance
    session = Session()
    
    try:
        # Check if stations table has data
        stations_count = session.query(Station).count()
        print(f"Number of stations in database: {stations_count}")

        # Check if charging_sessions table has data
        sessions_count = session.query(ChargingSession).count()
        print(f"Number of charging sessions in database: {sessions_count}")

        # If there are stations, show the first 3
        if stations_count > 0:
            print("\nFirst 3 stations:")
            stations = session.query(Station).limit(3).all()
            for station in stations:
                print(f"  - {station.name} ({station.latitude}, {station.longitude})")

    except Exception as e:
        print(f"Error checking database: {e}")
    finally:
        session.close()

if __name__ == "__main__":
    check_database()
