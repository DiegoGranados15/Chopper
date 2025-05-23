document.addEventListener('DOMContentLoaded', function() {
    const chatbox = document.getElementById('chatbox');
    const userInput = document.getElementById('userInput');
    const sendButton = document.getElementById('sendButton');
    const typingIndicator = document.getElementById('typingIndicator');
    
    // Event Listeners
    sendButton.addEventListener('click', sendMessage);
    userInput.addEventListener('keypress', function(e) {
        if (e.key === 'Enter') {
            sendMessage();
        }
    });
    
    // Funciones
    function scrollToBottom() {
        chatbox.scrollTop = chatbox.scrollHeight;
    }
    
    function getCurrentTime() {
        const now = new Date();
        return now.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
    }
    
    async function sendMessage() {
        const input = userInput.value.trim();
        if (!input) return;
        
        // Mostrar mensaje del usuario
        addMessage(input, 'user');
        userInput.value = '';
        
        // Mostrar indicador de typing
        typingIndicator.style.display = 'block';
        scrollToBottom();
        
        try {
            const response = await fetch("/chat", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ message: input }),
            });
            
            const data = await response.json();
            typingIndicator.style.display = 'none';
            addMessage(data.response, 'bot');
            
        } catch (error) {
            typingIndicator.style.display = 'none';
            addMessage("Lo siento, hubo un error al procesar tu solicitud. Por favor intenta nuevamente.", 'bot');
            console.error("Error:", error);
        }
    }
    
    function addMessage(text, sender) {
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${sender}-message`;
        messageDiv.innerHTML = `${text}<div class="timestamp">${getCurrentTime()}</div>`;
        chatbox.appendChild(messageDiv);
        scrollToBottom();
    }
    
    // Scroll inicial al fondo
    scrollToBottom();
});