import './ChatMessage.css'

function ChatMessage({ message }) {
  if (message.type === 'user') {
    return (
      <div className="message user-message">
        <div className="message-content">
          <p>{message.content}</p>
        </div>
      </div>
    )
  }

  // Bot message
  if (message.isError) {
    return (
      <div className="message bot-message">
        <div className="message-content">
          <div className="error-message">{message.content}</div>
        </div>
      </div>
    )
  }

  // Simple bot message (welcome message)
  if (message.content && !message.data) {
    return (
      <div className="message bot-message">
        <div className="message-content">
          <p>{message.content}</p>
          {message.hint && (
            <p className="message-hint">{message.hint}</p>
          )}
        </div>
      </div>
    )
  }

  // Bot message with data (query response)
  const data = message.data
  if (!data) return null

  const formatText = (text) => {
    if (!text) return ''
    return text
      .split('\n')
      .map((line, i) => (
        <span key={i}>
          {line.split(/(\*\*.*?\*\*|\*.*?\*)/).map((part, j) => {
            if (part.startsWith('**') && part.endsWith('**')) {
              return <strong key={j}>{part.slice(2, -2)}</strong>
            }
            if (part.startsWith('*') && part.endsWith('*') && part.length > 1) {
              return <em key={j}>{part.slice(1, -1)}</em>
            }
            return part
          })}
          {i < text.split('\n').length - 1 && <br />}
        </span>
      ))
  }

  return (
    <div className="message bot-message">
      <div className="message-content">
        {data.generated_answer ? (
          <div className="answer-section">
            {data.generated_answer.answer && (
              <div className="answer-text">
                {formatText(data.generated_answer.answer)}
              </div>
            )}
            
            {data.generated_answer.confidence !== undefined && (
              <span className="confidence-badge">
                Confidence: {Math.round(data.generated_answer.confidence * 100)}%
              </span>
            )}

            {data.generated_answer.timeline && data.generated_answer.timeline.length > 0 && (
              <div className="timeline-section">
                <div className="timeline-title">📅 Evolution Timeline</div>
                {data.generated_answer.timeline.map((entry, idx) => (
                  <div key={idx} className="timeline-item">
                    <div className="timeline-year">{entry.year}</div>
                    <div className="timeline-summary">
                      {formatText(entry.summary)}
                    </div>
                  </div>
                ))}
              </div>
            )}

            {data.generated_answer.sources && data.generated_answer.sources.length > 0 && (
              <div className="sources-section">
                <div className="sources-title">📚 Sources</div>
                {data.generated_answer.sources.slice(0, 5).map((source, idx) => (
                  <div key={idx} className="source-item">
                    <strong>{idx + 1}.</strong> {source.title || 'Unknown'} ({source.year || '?'})
                  </div>
                ))}
              </div>
            )}
          </div>
        ) : (
          <div className="answer-section">
            {data.results && data.results.length > 0 ? (
              <>
                <p>Here are the retrieved results:</p>
                {data.results.slice(0, 3).map((result, idx) => {
                  const text = result.text || ''
                  const preview = text.length > 200 ? text.substring(0, 200) + '...' : text
                  return (
                    <div key={idx} style={{ marginTop: '15px', padding: '12px', background: '#f5f5f5', borderRadius: '8px' }}>
                      <strong>Result {idx + 1}</strong> (Score: {result.score?.toFixed(3) || 'N/A'})<br />
                      <span style={{ fontSize: '14px', color: '#666' }}>{preview}</span>
                    </div>
                  )
                })}
              </>
            ) : (
              <p>No results found for your query.</p>
            )}
          </div>
        )}

        {data.query_entities && data.query_entities.length > 0 && (
          <div style={{ marginTop: '15px', paddingTop: '15px', borderTop: '1px solid #e0e0e0', fontSize: '12px', color: '#666' }}>
            📋 Detected entities: {data.query_entities.join(', ')}
          </div>
        )}
      </div>
    </div>
  )
}

export default ChatMessage

