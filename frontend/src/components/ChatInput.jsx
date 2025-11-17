import { useState, useRef, useEffect } from 'react'
import './ChatInput.css'

function ChatInput({ onSend, disabled, options, onOptionsChange }) {
  const [query, setQuery] = useState('')
  const textareaRef = useRef(null)

  useEffect(() => {
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto'
      textareaRef.current.style.height = textareaRef.current.scrollHeight + 'px'
    }
  }, [query])

  const handleSubmit = (e) => {
    e.preventDefault()
    if (query.trim() && !disabled) {
      onSend(query)
      setQuery('')
      if (textareaRef.current) {
        textareaRef.current.style.height = 'auto'
      }
    }
  }

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSubmit(e)
    }
  }

  const handleOptionChange = (key, value) => {
    onOptionsChange({
      ...options,
      [key]: value
    })
  }

  return (
    <div className="chat-input-container">
      <form onSubmit={handleSubmit} className="input-wrapper">
        <textarea
          ref={textareaRef}
          id="queryInput"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="Ask a question about research papers..."
          rows="1"
          disabled={disabled}
        />
        <button 
          type="submit"
          id="sendButton"
          disabled={disabled || !query.trim()}
        >
          <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <line x1="22" y1="2" x2="11" y2="13"></line>
            <polygon points="22 2 15 22 11 13 2 9 22 2"></polygon>
          </svg>
        </button>
      </form>
      <div className="input-options">
        <label className="checkbox-label">
          <input 
            type="checkbox" 
            checked={options.useGraph}
            onChange={(e) => handleOptionChange('useGraph', e.target.checked)}
          />
          <span>Use Graph</span>
        </label>
        <label className="checkbox-label">
          <input 
            type="checkbox" 
            checked={options.useGenerator}
            onChange={(e) => handleOptionChange('useGenerator', e.target.checked)}
          />
          <span>Generate Answer</span>
        </label>
        <select 
          value={options.maxHops}
          onChange={(e) => handleOptionChange('maxHops', parseInt(e.target.value))}
        >
          <option value="1">1 Hop</option>
          <option value="2">2 Hops</option>
          <option value="3">3 Hops</option>
        </select>
      </div>
    </div>
  )
}

export default ChatInput

