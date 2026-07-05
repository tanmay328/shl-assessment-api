from typing import List, Dict, Optional

class Evaluator:
    def __init__(self):
        self.metrics = {
            "recall_at_10": [],
            "groundedness": [],
            "accuracy": [],
            "response_times": []
        }
    
    def calculate_recall_at_10(self, recommendations: List[Dict], ground_truth: List[str]) -> float:
        """Calculate Recall@10"""
        if not ground_truth or not recommendations:
            return 0.0
        
        recommended_names = [r['name'] for r in recommendations[:10]]
        relevant_in_top = sum(1 for item in ground_truth if item in recommended_names)
        return relevant_in_top / len(ground_truth) if ground_truth else 0.0
    
    def check_groundedness(self, response: str, catalog_items: List[Dict]) -> float:
        """Check if response is grounded in catalog data"""
        if not catalog_items:
            return 1.0
        
        catalog_names = [item.get('name', '').lower() for item in catalog_items[:10]]
        response_lower = response.lower()
        
        mentions = sum(1 for name in catalog_names if name in response_lower)
        
        if mentions == 0:
            return 1.0
        
        words = set(response_lower.split())
        suspicious_words = [w for w in words if len(w) > 5 and w not in str(catalog_names)]
        
        grounded_ratio = mentions / (mentions + len(suspicious_words) + 1)
        return min(1.0, grounded_ratio)
    
    def log_metric(self, name: str, value: float):
        if name in self.metrics:
            self.metrics[name].append(value)
    
    def get_average_metrics(self) -> Dict:
        result = {}
        for key, values in self.metrics.items():
            if values:
                result[key] = sum(values) / len(values)
            else:
                result[key] = 0.0
        return result