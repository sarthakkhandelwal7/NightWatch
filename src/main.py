from fastapi import APIRouter, Depends
from typing import Dict
from ../
router = APIRouter(prefix='/invoke', tags=['Agent'])

@router.post('/orchestrator')
def invoke_agent(issue: Dict):
    