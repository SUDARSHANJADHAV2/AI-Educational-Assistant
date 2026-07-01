document.addEventListener('DOMContentLoaded', () => {
    const chatInput = document.getElementById('chat-input');
    const sendBtn = document.getElementById('send-btn');
    const chatContent = document.getElementById('chat-content');
    const addSourceModal = document.getElementById('add-source-modal');
    const closeModalBtn = document.getElementById('close-modal');
    
    // Suggestion chips
    const chips = document.querySelectorAll('.chip');
    chips.forEach(chip => {
        chip.addEventListener('click', () => {
            chatInput.value = chip.textContent;
            sendMessage();
        });
    });

    // Modal toggles
    document.querySelectorAll('.btn-secondary.w-full').forEach(btn => {
        btn.addEventListener('click', () => {
            addSourceModal.style.display = 'flex';
        });
    });

    closeModalBtn.addEventListener('click', () => {
        addSourceModal.style.display = 'none';
    });

    // Close modal on outside click
    addSourceModal.addEventListener('click', (e) => {
        if (e.target === addSourceModal) {
            addSourceModal.style.display = 'none';
        }
    });

    // Handle Enter key
    chatInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') {
            sendMessage();
        }
    });

    sendBtn.addEventListener('click', sendMessage);

    let conversationHistory = [
        { role: "system", content: "You are NotebookLM, a helpful, precise, and educational AI assistant. Answer questions clearly and concisely using proper markdown formatting." }
    ];

    async function sendMessage() {
        const text = chatInput.value.trim();
        if (!text) return;

        // Add user message to UI
        addMessageToUI('user', text);
        chatInput.value = '';
        
        // Hide welcome screen if it's there
        const welcome = document.querySelector('.chat-welcome');
        if (welcome) welcome.style.display = 'none';

        conversationHistory.push({ role: "user", content: text });

        // Add typing indicator
        const typingId = addTypingIndicator();
        scrollToBottom();

        try {
            // Check if LM Studio is running on localhost:1234
            const response = await fetch('http://localhost:1234/v1/chat/completions', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'Authorization': 'Bearer lm-studio'
                },
                body: JSON.stringify({
                    model: "local-model",
                    messages: conversationHistory,
                    temperature: 0.7,
                    max_tokens: 1024,
                    stream: true // Try streaming first
                })
            });

            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            // Remove typing indicator
            document.getElementById(typingId).remove();

            // Setup new message bubble for assistant
            const msgId = 'msg-' + Date.now();
            const msgEl = addMessageToUI('assistant', '', msgId);
            const contentEl = msgEl.querySelector('.message-bubble');
            
            // Handle streaming response
            const reader = response.body.getReader();
            const decoder = new TextDecoder("utf-8");
            let assistantResponse = "";

            while (true) {
                const { done, value } = await reader.read();
                if (done) break;
                
                const chunk = decoder.decode(value, { stream: true });
                const lines = chunk.split('\n');
                
                for (const line of lines) {
                    if (line.startsWith('data: ') && line !== 'data: [DONE]') {
                        try {
                            const data = JSON.parse(line.substring(6));
                            if (data.choices && data.choices[0].delta && data.choices[0].delta.content) {
                                assistantResponse += data.choices[0].delta.content;
                                // Basic markdown to HTML (simplified)
                                contentEl.innerHTML = formatMarkdown(assistantResponse);
                                scrollToBottom();
                            }
                        } catch (e) {
                            console.error('Error parsing chunk:', e);
                        }
                    }
                }
            }

            conversationHistory.push({ role: "assistant", content: assistantResponse });

        } catch (error) {
            console.error('Error:', error);
            document.getElementById(typingId).remove();
            
            addMessageToUI('assistant', '⚠️ Cannot connect to LM Studio. Please make sure LM Studio is running on your computer, you have loaded a model, and the Local Inference Server is started on port 1234.');
        }
    }

    function addMessageToUI(role, text, id = null) {
        const msgDiv = document.createElement('div');
        msgDiv.className = `message ${role}`;
        if (id) msgDiv.id = id;

        const bubble = document.createElement('div');
        bubble.className = 'message-bubble';
        bubble.innerHTML = formatMarkdown(text);
        
        msgDiv.appendChild(bubble);
        chatContent.appendChild(msgDiv);
        scrollToBottom();
        
        return msgDiv;
    }

    function addTypingIndicator() {
        const id = 'typing-' + Date.now();
        const msgDiv = document.createElement('div');
        msgDiv.className = 'message assistant';
        msgDiv.id = id;

        const indicator = document.createElement('div');
        indicator.className = 'typing-indicator message-bubble';
        indicator.innerHTML = `
            <div class="typing-dot"></div>
            <div class="typing-dot"></div>
            <div class="typing-dot"></div>
        `;
        
        msgDiv.appendChild(indicator);
        chatContent.appendChild(msgDiv);
        return id;
    }

    function scrollToBottom() {
        chatContent.scrollTop = chatContent.scrollHeight;
    }

    // Very basic markdown parser for the UI
    function formatMarkdown(text) {
        if (!text) return '';
        
        // Headers
        let html = text.replace(/^### (.*$)/gim, '<h3>$1</h3>')
                       .replace(/^## (.*$)/gim, '<h2>$1</h2>')
                       .replace(/^# (.*$)/gim, '<h1>$1</h1>');
        
        // Bold
        html = html.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
        
        // Italic
        html = html.replace(/\*(.*?)\*/g, '<em>$1</em>');
        
        // Code blocks
        html = html.replace(/```([\s\S]*?)```/g, '<pre><code>$1</code></pre>');
        
        // Inline code
        html = html.replace(/`(.*?)`/g, '<code>$1</code>');
        
        // Line breaks
        html = html.replace(/\n/g, '<br>');
        
        // Fix empty breaks
        html = html.replace(/<br><br>/g, '</p><p>');
        html = '<p>' + html + '</p>';
        html = html.replace(/<p><\/p>/g, '');
        
        return html;
    }
});
