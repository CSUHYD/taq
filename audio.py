#!/usr/bin/env python3
"""
自动播放新生成的Question
监控web服务中的新问题并通过机器人TTS播放
"""

import time
import requests
from unitree_sdk2py.core.channel import ChannelFactoryInitialize
from unitree_sdk2py.g1.audio.g1_audio_client import AudioClient
from unitree_sdk2py.g1.loco.g1_loco_client import LocoClient

class QuestionAudioPlayer:
    def __init__(self, network_interface='enp3s0', server_url="http://localhost:5050"):
        self.server_url = server_url
        self.seen_questions = set()
        
        # 初始化机器人连接
        ChannelFactoryInitialize(0, network_interface)
        
        self.audio_client = AudioClient()  
        self.audio_client.SetTimeout(10.0)
        self.audio_client.Init()
        
        self.sport_client = LocoClient()  
        self.sport_client.SetTimeout(10.0)
        self.sport_client.Init()
        
        # 设置音量
        self.audio_client.SetVolume(85)
        print(f"Audio volume set to: {self.audio_client.GetVolume()}")
        
    def get_conversation_history(self):
        """获取对话历史"""
        try:
            response = requests.get(f"{self.server_url}/conversation")
            if response.status_code == 200:
                return response.json()
            else:
                return []
        except requests.exceptions.RequestException:
            return []
    
    def extract_speech_content(self, conversation):
        """从对话历史中提取需要播放的内容"""
        speech_items = []
        for message in conversation:
            if message.get("type") == "robot" and "question" in message:
                speech_data = {
                    "timestamp": message.get("timestamp"),
                    "content": message.get("question"),
                    "type": "question"
                }
                speech_items.append(speech_data)
            elif message.get("type") == "robot_response" and "robot_reply" in message:
                speech_data = {
                    "timestamp": message.get("timestamp"),
                    "content": message.get("robot_reply"),
                    "type": "reply"
                }
                speech_items.append(speech_data)
        return speech_items
    
    def play_new_speech_content(self, speech_items):
        """播放新的语音内容（问题和回复）"""
        for item in speech_items:
            # 使用时间戳和内容创建唯一标识
            content_id = f"{item['timestamp']}_{item['content'][:50]}"
            
            if content_id not in self.seen_questions:
                self.seen_questions.add(content_id)
                
                if item['type'] == 'question':
                    print(f"Playing question: {item['content']}")
                elif item['type'] == 'reply':
                    print(f"Playing reply: {item['content']}")
                
                # 使用TTS播放内容
                self.audio_client.TtsMaker(item['content'], 0)
                
                # 等待播放完成（估算时间：每个字符约0.1秒）
                play_time = max(3, len(item['content']) * 0.1)
                time.sleep(play_time)
    
    def monitor_and_play(self, interval=2):
        """监控并播放新的语音内容"""
        print(f"Starting speech audio monitor...")
        print(f"Server: {self.server_url}")
        print(f"Check interval: {interval} seconds")
        print("Press Ctrl+C to stop")
        
        # 初始化欢迎语
        self.audio_client.TtsMaker("语音播放系统已启动", 0)
        time.sleep(3)
        
        while True:
            try:
                conversation = self.get_conversation_history()
                speech_items = self.extract_speech_content(conversation)
                
                if speech_items:
                    self.play_new_speech_content(speech_items)
                
                time.sleep(interval)
                
            except KeyboardInterrupt:
                print("\nStopping audio monitor...")
                self.audio_client.TtsMaker("语音播放系统已停止", 0)
                time.sleep(3)
                break
            except Exception:
                time.sleep(interval)

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Monitor and play robot questions and replies via TTS")
    parser.add_argument("--network_interface",  default="enp3s0", help="Network interface for robot connection")
    parser.add_argument("--url", default="http://localhost:5050", 
                       help="Server URL (default: http://localhost:5050)")
    parser.add_argument("--interval", type=int, default=2,
                       help="Check interval in seconds (default: 2)")
    
    args = parser.parse_args()
    
    player = QuestionAudioPlayer(args.network_interface, args.url)
    player.monitor_and_play(args.interval)

if __name__ == "__main__":
    main()
    