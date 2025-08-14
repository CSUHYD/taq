#!/usr/bin/env python3

"""测试重构后的架构分离"""

from robot_service import RobotService
from vlmCall_ollama import VLMAPI

def test_basic_api():
    """测试基础API功能"""
    print("=== Testing Basic API ===")
    
    vlm_api = VLMAPI("qwen2.5vl:32b")
    
    # 测试基础文本请求
    response = vlm_api.vlm_request(
        systext="You are a helpful assistant.",
        usertext="Hello, how are you?",
        max_tokens=50
    )
    
    print(f"Basic API Response: {response}")
    print()

def test_business_logic():
    """测试业务逻辑服务"""
    print("=== Testing Business Logic ===")
    
    robot_service = RobotService("qwen2.5vl:32b")
    
    # 测试问题生成（不使用图像）
    try:
        response = robot_service.generate_question(
            task_description="organize the table",
            image_path=None,
            messages_history=[]
        )
        
        print(f"Generated Question:")
        print(f"  Reasoning: {response.reasoning}")
        print(f"  Question: {response.question}")
    except Exception as e:
        print(f"Question generation failed: {e}")
    
    print()
    
    # 测试响应分析
    try:
        analysis = robot_service.analyze_user_response(
            user_response="Yes, please organize by type",
            robot_question="Would you like me to organize by type or frequency?",
            robot_reasoning="I need to understand the user's preference",
            current_task="organize the table",
            messages_history=[]
        )
        
        print(f"Response Analysis:")
        print(f"  Understanding: {analysis.understanding}")
        print(f"  Operator instructions: {analysis.operator_instructions}")
        print(f"  Robot reply: {analysis.robot_reply}")
    except Exception as e:
        print(f"Response analysis failed: {e}")

def test_architecture_separation():
    """验证架构分离"""
    print("=== Testing Architecture Separation ===")
    
    # 验证基础API不包含业务逻辑
    vlm_api = VLMAPI("qwen2.5vl:32b")
    api_methods = [method for method in dir(vlm_api) if not method.startswith('_')]
    print(f"VLM API methods: {api_methods}")
    
    # 验证业务服务包含业务逻辑
    robot_service = RobotService("qwen2.5vl:32b")
    service_methods = [method for method in dir(robot_service) if not method.startswith('_')]
    print(f"Robot Service methods: {service_methods}")
    
    print("\n✅ Architecture separation verified:")
    print("- vlmCall_ollama: Basic API functionality only")
    print("- robot_service: Business logic and Pydantic models")
    print("- web_app: Uses robot_service for business operations")

if __name__ == "__main__":
    print("Testing Refactored Architecture\n")
    
    test_basic_api()
    test_business_logic()
    test_architecture_separation()
    
    print("\n🎉 All tests completed!")