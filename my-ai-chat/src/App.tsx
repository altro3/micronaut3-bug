import {useEffect, useRef, useState} from 'react';
import Sidebar from './components/sidebar/Sidebar';
import ChatMessage from './components/chat/ChatMessage';
import ChatLoader from './components/chat/ChatLoader';
import ChatInput from './components/chat/ChatInput';
import {sendChatCompletionStream} from './api/chatClient';
import type {Message, OrchestratorStatus} from './types/chat';

export default function App() {
    const [messages, setMessages] = useState<Message[]>([
        {id: '1', role: 'assistant', content: 'Привет! Я твой локальный ИИ. Мой оркестратор готов к работе.'}
    ]);
    const [isLoading, setIsLoading] = useState(false);
    const [status, setStatus] = useState<OrchestratorStatus>('connected');

    const messagesEndRef = useRef<HTMLDivElement>(null);

    useEffect(() => {
        messagesEndRef.current?.scrollIntoView({behavior: 'smooth'});
    }, [messages, isLoading]);

    const handleSendMessage = async (text: string) => {
        if (isLoading) return;

        const userMessage: Message = {
            id: Date.now().toString(),
            role: 'user',
            content: text,
        };

        const assistantMessageId = (Date.now() + 1).toString();
        const updatedMessages = [...messages, userMessage];

        setMessages(updatedMessages);
        setIsLoading(true);

        try {
            // Передаем логику кумулятивной замены контента
            await sendChatCompletionStream(updatedMessages, (fullCleanText) => {
                setMessages((prev) => {
                    const exists = prev.some(m => m.id === assistantMessageId);
                    if (!exists) {
                        return [
                            ...prev,
                            { id: assistantMessageId, role: 'assistant', content: fullCleanText }
                        ];
                    }
                    return prev.map(m =>
                        m.id === assistantMessageId ? { ...m, content: fullCleanText } : m
                    );
                });
            });

            setStatus('connected');
        } catch (error) {
            console.error('Stream error:', error);
            setStatus('error');
        } finally {
            setIsLoading(false);
        }
    };

    return (
        <div className="flex h-screen bg-gray-900 text-gray-100 font-sans">
            <Sidebar status={status}/>

            <div className="flex-1 flex flex-col h-full bg-gray-900">
                <div className="h-16 border-b border-gray-800 px-8 flex items-center justify-between bg-gray-900/50 backdrop-blur">
                    <div className="font-medium">Локальная сессия ассистента</div>
                </div>

                <div className="flex-1 overflow-y-auto p-8 space-y-6">
                    {messages.map((msg) => (
                        <ChatMessage key={msg.id} message={msg}/>
                    ))}
                    {isLoading && <ChatLoader/>}
                    <div ref={messagesEndRef}/>
                </div>

                <ChatInput onSendMessage={handleSendMessage} isLoading={isLoading}/>
            </div>
        </div>
    );
}
