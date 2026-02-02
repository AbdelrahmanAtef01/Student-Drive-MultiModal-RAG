from typing import TypedDict, List, Any, Optional

class GraphState(TypedDict):
    file_id: str
    file_path: str
    file_type: str            
    
    # Data containers
    layout_results: List[Any]  
    rag_ready_data: List[Any]  
    
    # Status
    status: str
    error: Optional[str]