'use client'

import { useState } from 'react'
import styles from './page.module.css'

export default function Home() {
  const [query, setQuery] = useState('')
  const [messages, setMessages] = useState<Array<{role: string, content: string}>>([])
  const [loading, setLoading] = useState(false)

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    if (!query.trim()) return

    const userMessage = query
    setQuery('')
    setMessages(prev => [...prev, { role: 'user', content: userMessage }])
    setLoading(true)

    try {
      const apiUrl = process.env.API_URL || 'http://localhost:8000'
      const response = await fetch(`${apiUrl}/rag`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query: userMessage }),
      })

      if (!response.ok) throw new Error('API request failed')

      const data = await response.json()
      setMessages(prev => [...prev, { role: 'assistant', content: data.answer }])
    } catch (error) {
      console.error('Error:', error)
      setMessages(prev => [...prev, {
        role: 'assistant',
        content: 'Sorry, something went wrong. Please try again.'
      }])
    } finally {
      setLoading(false)
    }
  }

  return (
    <main className={styles.main}>
      <div className={styles.container}>
        <h1 className={styles.title}>RAG System</h1>
        <p className={styles.subtitle}>Ask me anything</p>

        <div className={styles.chat}>
          {messages.length === 0 && (
            <div className={styles.empty}>
              Start a conversation by asking a question below
            </div>
          )}
          {messages.map((msg, idx) => (
            <div
              key={idx}
              className={msg.role === 'user' ? styles.userMessage : styles.assistantMessage}
            >
              <div className={styles.messageRole}>
                {msg.role === 'user' ? 'You' : 'AI'}
              </div>
              <div className={styles.messageContent}>
                {msg.content}
              </div>
            </div>
          ))}
          {loading && (
            <div className={styles.assistantMessage}>
              <div className={styles.messageRole}>AI</div>
              <div className={styles.messageContent}>
                <div className={styles.loading}>Thinking...</div>
              </div>
            </div>
          )}
        </div>

        <form onSubmit={handleSubmit} className={styles.form}>
          <input
            type="text"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Type your question..."
            className={styles.input}
            disabled={loading}
          />
          <button
            type="submit"
            className={styles.button}
            disabled={loading || !query.trim()}
          >
            Send
          </button>
        </form>
      </div>
    </main>
  )
}
