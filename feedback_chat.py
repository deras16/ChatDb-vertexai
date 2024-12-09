import json
from datetime import datetime
import os
from typing import Dict, Any

class ChatEvaluationStorage:
    def __init__(self, storage_dir: str = "evaluation_data"):
        self.storage_dir = storage_dir
        os.makedirs(storage_dir, exist_ok=True)
        self.storage_file = os.path.join(storage_dir, "chat_evaluation.json")
        
        if not os.path.exists(self.storage_file):
            initial_data = {
                "interactions": []
            }
            self._save_data(initial_data)
    
    def _load_data(self) -> Dict[str, Any]:
        try:
            with open(self.storage_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading data: {e}")
            return {"interactions": []}
    
    def _save_data(self, data: Dict[str, Any]) -> None:
        with open(self.storage_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    
    def save_interaction(self, user_query: str, sql_query: str, ai_response: str) -> int:
        data = self._load_data()
        interaction_id = len(data["interactions"]) + 1
        
        interaction = {
            "interaction_id": interaction_id,
            "timestamp": datetime.now().isoformat(),
            "user_query": user_query,
            "sql_query": sql_query,
            "ai_response": ai_response,
            "feedback": None
        }
        
        data["interactions"].append(interaction)
        self._save_data(data)
        return interaction_id
    
    def update_feedback(self, interaction_id: int, feedback_data: Dict[str, bool]) -> bool:
        """
        Actualiza el feedback de una interacción
        
        Args:
            interaction_id: ID de la interacción
            feedback_data: Diccionario que contiene {'thumbs_up': bool}
        """
        data = self._load_data()
        
        for interaction in data["interactions"]:
            if interaction["interaction_id"] == interaction_id:
                # El feedback de streamlit_feedback con type="thumbs" devuelve {'thumbs_up': True/False}
                interaction["feedback"] = {
                    "is_positive": feedback_data.get('thumbs_up', None),
                    "timestamp": datetime.now().isoformat()
                }
                
                self._save_data(data)
                return True
        return False