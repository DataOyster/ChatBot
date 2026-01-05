cd backend
start cmd /k "uvicorn main:app --reload"
cd ../frontend
start cmd /k "npm run dev"
start http://localhost:3000
