#!/usr/bin/env python3
"""
Test script to verify the timer and logging system functionality
"""

import os
import json
import time
from datetime import datetime
from robot_service import RobotService

def test_timer_logging_system():
    print("Testing Timer and Logging System...")
    print("=" * 50)
    
    # Initialize robot service
    robot_service = RobotService()
    
    # Test 1: Start logging session with custom ID
    print("Test 1: Starting session with custom ID...")
    custom_session_id = "test_timer_001"
    session_id = robot_service.start_logging_session("organize_desktop", custom_session_id)
    print(f"Session started: {session_id}")
    
    # Verify session naming convention
    expected_prefix = "organize_desktop_test_timer_001_"
    if session_id.startswith(expected_prefix):
        print("✓ Session naming convention correct")
    else:
        print("✗ Session naming issue")
    
    # Test 2: Log timing data
    print("\nTest 2: Logging timing data...")
    timing_data = {
        "start_time": "2024-01-01T10:00:00.000Z",
        "end_time": "2024-01-01T10:05:30.000Z", 
        "total_time_seconds": 330,
        "interaction_times": [
            {"interaction_type": "question_generation_started", "timestamp": "2024-01-01T10:00:15.000Z"},
            {"interaction_type": "question_generation_completed", "timestamp": "2024-01-01T10:00:45.000Z"},
            {"interaction_type": "response_analysis_started", "timestamp": "2024-01-01T10:01:00.000Z"},
            {"interaction_type": "response_analysis_completed", "timestamp": "2024-01-01T10:01:15.000Z"}
        ]
    }
    
    robot_service.log_timing_data(timing_data)
    print("✓ Timing data logged successfully")
    
    # Test 3: Verify session directory and files
    print("\nTest 3: Verifying session files...")
    session_dir = os.path.join("logs/experiments", session_id)
    
    if os.path.exists(session_dir):
        print(f"✓ Session directory created: {session_dir}")
        
        # Check for images directory
        images_dir = os.path.join(session_dir, "images")
        if os.path.exists(images_dir):
            print("✓ Images directory created")
        
        # Check for session log file
        log_file = os.path.join(session_dir, "session_log.json")
        if os.path.exists(log_file):
            print("✓ Session log file created")
            
            # Verify log contents
            with open(log_file, 'r', encoding='utf-8') as f:
                log_data = json.load(f)
                
            print(f"  - Session ID: {log_data.get('session_id')}")
            print(f"  - Custom Session ID: {log_data.get('custom_session_id')}")
            print(f"  - Task: {log_data.get('task_description')}")
            print(f"  - Language: {log_data.get('language')}")
            print(f"  - Timing data present: {'timing_data' in log_data}")
            
            if 'timing_data' in log_data:
                timing = log_data['timing_data']
                print(f"  - Task start time: {timing.get('task_start_time')}")
                print(f"  - Task end time: {timing.get('task_end_time')}")
                print(f"  - Total time: {timing.get('total_task_time_seconds')}s")
                print(f"  - Interaction count: {len(timing.get('interaction_timings', []))}")
                
                if timing.get('total_task_time_seconds') == 330:
                    print("✓ Timing data correctly stored")
                else:
                    print("✗ Timing data mismatch")
        else:
            print("✗ Session log file not found")
    else:
        print("✗ Session directory not created")
    
    # Test 4: End session
    print("\nTest 4: Ending session...")
    ended_session = robot_service.end_logging_session()
    print(f"Session ended: {ended_session}")
    
    # Test 5: Verify final log structure
    print("\nTest 5: Verifying final log structure...")
    if os.path.exists(log_file):
        with open(log_file, 'r', encoding='utf-8') as f:
            final_log_data = json.load(f)
            
        required_fields = ['session_id', 'custom_session_id', 'start_time', 'end_time', 
                          'task_description', 'language', 'interactions', 'timing_data']
        
        missing_fields = [field for field in required_fields if field not in final_log_data]
        
        if not missing_fields:
            print("✓ All required fields present in final log")
        else:
            print(f"✗ Missing fields: {missing_fields}")
            
        # Check timing data structure
        if 'timing_data' in final_log_data:
            timing_structure = final_log_data['timing_data']
            expected_timing_fields = ['task_start_time', 'task_end_time', 'total_task_time_seconds', 'interaction_timings']
            missing_timing = [field for field in expected_timing_fields if field not in timing_structure]
            
            if not missing_timing:
                print("✓ Complete timing data structure")
            else:
                print(f"✗ Missing timing fields: {missing_timing}")
    
    print("\n" + "=" * 50)
    print("Timer and Logging System Test Complete!")
    
    # Display summary
    print(f"\nSession Summary:")
    print(f"Session ID: {session_id}")
    print(f"Log Directory: {session_dir}")
    print(f"Total Files Created: {len(os.listdir(session_dir)) if os.path.exists(session_dir) else 0}")

if __name__ == "__main__":
    test_timer_logging_system()
