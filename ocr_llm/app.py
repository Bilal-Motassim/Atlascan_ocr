# app.py
from fastapi import FastAPI
from services import id_card_service, passport_service, drivers_license_service

app = FastAPI()

# Register endpoints from services
app.include_router(id_card_service.router, prefix="/id-card")
app.include_router(passport_service.router, prefix="/passport")
app.include_router(drivers_license_service.router, prefix="/drivers-license")
