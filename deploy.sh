#!/bin/bash

# TalkBot - Automated Deployment Script
set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
NC='\033[0m'

# Logging functions
print_status() { echo -e "${BLUE}[INFO]${NC} $1"; }
print_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
print_warning() { echo -e "${YELLOW}[WARNING]${NC} $1"; }
print_error() { echo -e "${RED}[ERROR]${NC} $1"; }
print_step() { echo -e "${PURPLE}[STEP]${NC} $1"; }

# Utility functions
command_exists() { command -v "$1" >/dev/null 2>&1; }

# Main functions
install_ollama() {
    print_step "Installing Ollama..."
    
    if command_exists ollama; then
        print_success "Ollama already installed"
        return 0
    fi
    
    print_status "Installing Ollama..."
    if [[ "$OSTYPE" == "darwin"* ]]; then
        if command_exists brew; then
            brew install ollama
        else
            /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
            export PATH="/opt/homebrew/bin:$PATH"
            brew install ollama
        fi
    elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
        curl -fsSL https://ollama.ai/install.sh | sh
    else
        print_error "Unsupported OS: $OSTYPE"
        return 1
    fi
    
    if command_exists ollama; then
        print_success "Ollama installed successfully"
    else
        print_warning "Ollama installation failed, continuing without it"
        return 1
    fi
}

start_ollama() {
    print_step "Starting Ollama service..."
    
    if ! command_exists ollama; then
        print_warning "Ollama not installed, skipping"
        return 1
    fi
    
    if curl -s http://localhost:11434/api/tags >/dev/null 2>&1; then
        print_success "Ollama service already running"
        return 0
    fi
    
    print_status "Starting Ollama service..."
    nohup ollama serve > /dev/null 2>&1 &
    
    # Wait for service to start
    for i in {1..30}; do
        if curl -s http://localhost:11434/api/tags >/dev/null 2>&1; then
            print_success "Ollama service started"
            return 0
        fi
        sleep 1
    done
    
    print_warning "Failed to start Ollama service"
    return 1
}

setup_environment() {
    print_step "Setting up environment..."
    
    if [ ! -f .env ]; then
        if [ -f .env.sample ]; then
            print_status "Creating .env from .env.sample..."
            cp .env.sample .env
            print_success ".env file created from template"
            print_warning "Please edit .env file with your configuration before continuing"
            echo ""
            print_status "Required configuration:"
            echo "  - OPENAI_API_KEY: Your OpenAI API key"
            echo "  - VHYS_FASTER_WHISPER_MODEL: ASR model (tiny.en, base.en, etc.)"
            echo "  - VHYS_PIPER_MODEL_PATH: Path to TTS model"
            echo ""
            read -p "Press Enter after configuring .env file..."
        else
            print_error ".env.sample not found! Please create it first."
            return 1
        fi
    else
        print_success ".env file already exists"
    fi
    
    # Validate required environment variables
    if ! grep -q "OPENAI_API_KEY=" .env || grep -q "your_openai_api_key_here" .env; then
        print_warning "OPENAI_API_KEY not configured in .env file"
        print_status "Please set your OpenAI API key in .env file"
    fi
}

install_dependencies() {
    print_step "Installing Python dependencies..."
    
    # Create virtual environment if needed
    if [ ! -d ".venv" ]; then
        print_status "Creating virtual environment..."
        python3 -m venv .venv
    fi
    
    # Activate and install
    source .venv/bin/activate
    pip install --upgrade pip
    pip install -r requirements.txt
    print_success "Python dependencies installed"
}

install_node_dependencies() {
    print_step "Installing Node.js dependencies..."
    
    if [ ! -d "client/node_modules" ]; then
        cd client && npm install && cd ..
        print_success "Node.js dependencies installed"
    else
        print_success "Node.js dependencies already installed"
    fi
}

setup_models() {
    print_step "Setting up AI models..."
    
    source .venv/bin/activate
    python3 setup_models.py
    
    if [ $? -eq 0 ]; then
        print_success "Models setup completed"
    else
        print_warning "Some models failed to setup, but continuing..."
    fi
}

start_development() {
    print_step "Starting development servers..."
    
    source .venv/bin/activate
    
    # Start backend
    print_status "Starting backend server..."
    python3 -m uvicorn server.main:app --host 0.0.0.0 --port 8080 --reload &
    BACKEND_PID=$!
    
    # Wait for backend
    sleep 5
    if curl -s http://localhost:8080/health >/dev/null 2>&1; then
        print_success "Backend server started"
    else
        print_warning "Backend server may not be ready yet"
    fi
    
    # Start frontend
    print_status "Starting frontend server..."
    cd client && npm run dev &
    FRONTEND_PID=$!
    cd ..
    
    sleep 3
    print_success "Development servers started!"
    echo ""
    print_status "🎉 TalkBot is now running!"
    print_status "Frontend: http://localhost:5173"
    print_status "Backend: http://localhost:8080"
    echo ""
    print_status "Press Ctrl+C to stop servers..."
    
    # Cleanup function
    cleanup() {
        print_status "Stopping servers..."
        kill $BACKEND_PID $FRONTEND_PID 2>/dev/null || true
        print_success "Servers stopped"
        exit 0
    }
    
    trap cleanup INT TERM
    wait
}

start_docker() {
    print_step "Starting with Docker Compose..."
    
    docker-compose down 2>/dev/null || true
    docker-compose up --build -d
    
    sleep 10
    if curl -s http://localhost:8080/health >/dev/null 2>&1; then
        print_success "Docker deployment complete!"
        print_status "Frontend: http://localhost"
        print_status "Backend: http://localhost:8080"
    else
        print_warning "Services may still be starting up..."
    fi
    
    echo ""
    print_status "Showing logs (Ctrl+C to exit)..."
    docker-compose logs -f
}

# Main execution
main() {
    echo "🚀 TalkBot - Automated Deployment"
    echo "======================================="
    
    # Create directories
    mkdir -p metrics models
    
    # Setup
    setup_environment
    install_dependencies
    install_node_dependencies
    install_ollama
    start_ollama
    setup_models
    
    # Ask for deployment method
    echo ""
    print_status "Deployment ready! Choose your method:"
    echo "1) Local development (recommended for testing)"
    echo "2) Docker Compose (recommended for production)"
    echo "3) Exit"
    echo ""
    read -p "Enter your choice (1-3): " choice
    
    case $choice in
        1) start_development ;;
        2) start_docker ;;
        3) print_status "Exiting..."; exit 0 ;;
        *) print_error "Invalid choice!"; exit 1 ;;
    esac
}

# Run main function
main