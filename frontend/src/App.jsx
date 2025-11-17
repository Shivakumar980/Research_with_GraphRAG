import { useState, useRef, useEffect } from 'react'
import ChatMessage from './components/ChatMessage'
import ChatInput from './components/ChatInput'
import Header from './components/Header'
import './App.css'

const API_URL = '/api/query'

function App() {
  const [messages, setMessages] = useState([
    {
      type: 'bot',
      content: '👋 Hello! I\'m your GraphRAG research assistant. I can help you explore research papers using knowledge graphs and semantic search.',
      hint: 'Try asking: "How did attention mechanisms evolve from Transformers to Flash Attention?"'
    }
  ])
  const [isLoading, setIsLoading] = useState(false)
  const [options, setOptions] = useState({
    useGraph: true,
    useGenerator: true,
    maxHops: 2
  })
  const messagesEndRef = useRef(null)

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }

  useEffect(() => {
    scrollToBottom()
  }, [messages])

  const sendMessage = async (query) => {
    if (!query.trim() || isLoading) return

    // Add user message
    setMessages(prev => [...prev, { type: 'user', content: query }])
    setIsLoading(true)

    try {
      const response = await fetch(API_URL, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          query: query,
          use_graph: options.useGraph,
          use_generator: options.useGenerator,
          max_hops: options.maxHops,
          top_k: 5
        })
      })

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`)
      }

      const data = await response.json()
      
      // Add bot response
      setMessages(prev => [...prev, {
        type: 'bot',
        data: data
      }])

    } catch (error) {
      console.error('Error:', error)
      setMessages(prev => [...prev, {
        type: 'bot',
        content: 'Sorry, I encountered an error. Please try again.',
        isError: true
      }])
    } finally {
      setIsLoading(false)
    }
  }

  return (
    <div className="container">
      <Header />
      <div className="chat-container">
        <div className="chat-messages">
          {messages.map((message, index) => (
            <ChatMessage key={index} message={message} />
          ))}
          {isLoading && (
            <div className="message bot-message">
              <div className="message-content">
                <div className="typing-indicator">
                  <div className="typing-dot"></div>
                  <div className="typing-dot"></div>
                  <div className="typing-dot"></div>
                </div>
              </div>
            </div>
          )}
          <div ref={messagesEndRef} />
        </div>
        <ChatInput 
          onSend={sendMessage} 
          disabled={isLoading}
          options={options}
          onOptionsChange={setOptions}
        />
      </div>
    </div>
  )
}

export default App

