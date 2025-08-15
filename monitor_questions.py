#!/usr/bin/env python3
"""
实时监控并打印最新的Question
通过轮询web服务获取对话历史，并实时显示新的问题
"""

import requests
import time
from datetime import datetime

class QuestionMonitor:
    def __init__(self, server_url="http://localhost:5050"):
        self.server_url = server_url
        self.last_question_count = 0
        self.seen_questions = set()
        
    def get_conversation_history(self):
        """获取对话历史"""
        try:
            response = requests.get(f"{self.server_url}/conversation")
            if response.status_code == 200:
                return response.json()
            else:
                print(f"Error fetching conversation: {response.status_code}")
                return []
        except requests.exceptions.RequestException as e:
            print(f"Connection error: {e}")
            return []
    
    def extract_questions(self, conversation):
        """从对话历史中提取问题"""
        questions = []
        for message in conversation:
            if message.get("type") == "robot" and "question" in message:
                question_data = {
                    "timestamp": message.get("timestamp"),
                    "question": message.get("question"),
                    "reasoning": message.get("reasoning", "")
                }
                questions.append(question_data)
        return questions
    
    def print_new_questions(self, questions):
        """打印新的问题"""
        for q in questions:
            # 使用时间戳和问题内容创建唯一标识
            question_id = f"{q['timestamp']}_{q['question'][:50]}"
            
            if question_id not in self.seen_questions:
                self.seen_questions.add(question_id)
                print(q['question'])
    
    def monitor(self, interval=2):
        """开始监控"""
        while True:
            try:
                conversation = self.get_conversation_history()
                questions = self.extract_questions(conversation)
                
                if questions:
                    self.print_new_questions(questions)
                
                time.sleep(interval)
                
            except KeyboardInterrupt:
                break
            except Exception as e:
                time.sleep(interval)

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Monitor and print latest questions from robot service")
    parser.add_argument("--url", default="http://localhost:5050", 
                       help="Server URL (default: http://localhost:5050)")
    parser.add_argument("--interval", type=int, default=2,
                       help="Check interval in seconds (default: 2)")
    
    args = parser.parse_args()
    
    monitor = QuestionMonitor(args.url)
    monitor.monitor(args.interval)

if __name__ == "__main__":
    main()