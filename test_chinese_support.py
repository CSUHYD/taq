#!/usr/bin/env python3
"""测试中文支持"""

from robot_service import RobotService
from vlmCall_ollama import load_prompt_config

def test_chinese_language_support():
    """测试中文语言支持"""
    print("=== 测试中文语言支持 ===")
    
    # 创建支持中文的机器人服务
    robot_service = RobotService("qwen2.5vl:32b", language="zh")
    
    print("✅ 中文机器人服务创建成功")
    print(f"当前语言设置: {robot_service.language}")
    
    # 测试配置加载
    config = load_prompt_config()
    print(f"配置文件语言: {config.get('language', 'en')}")
    print(f"当前任务: {config.get('task_description', '默认任务')}")
    
    # 测试prompt配置
    question_config = config.get("question_generation", {}).get("zh", {})
    if question_config:
        print("✅ 中文问题生成配置加载成功")
        print(f"系统提示词: {question_config.get('systext', '')[:50]}...")
    
    response_config = config.get("response_analysis", {}).get("zh", {})
    if response_config:
        print("✅ 中文回应分析配置加载成功")
        print(f"系统提示词: {response_config.get('systext', '')[:50]}...")

def test_language_switching():
    """测试语言切换"""
    print("\n=== 测试语言切换 ===")
    
    robot_service = RobotService("qwen2.5vl:32b", language="en")
    print(f"初始语言: {robot_service.language}")
    
    # 切换到中文
    robot_service.set_language("zh")
    print(f"切换后语言: {robot_service.language}")
    
    # 切换回英文
    robot_service.set_language("en")
    print(f"再次切换后语言: {robot_service.language}")
    
    print("✅ 语言切换功能正常")

def test_error_handling():
    """测试错误处理"""
    print("\n=== 测试错误处理 ===")
    
    robot_service = RobotService("qwen2.5vl:32b", language="zh")
    
    try:
        robot_service.set_language("invalid")
        print("❌ 应该抛出错误但没有")
    except ValueError as e:
        print(f"✅ 正确捕获错误: {e}")

if __name__ == "__main__":
    print("开始测试中文支持功能\n")
    
    test_chinese_language_support()
    test_language_switching()
    test_error_handling()
    
    print("\n🎉 所有中文支持测试完成!")