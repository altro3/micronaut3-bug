import {useEffect, useRef, useState} from 'react';
import Sidebar from './components/sidebar/Sidebar';
import ChatMessage from './components/chat/ChatMessage';
import ChatLoader from './components/chat/ChatLoader';
import ChatInput from './components/chat/ChatInput';
import {sendChatCompletionStream} from './api/chatClient';
// Предполагаем, что вы добавите функцию инициализации в ваш chatClient.ts
import {initOrchestratorSession} from './api/chatClient';
import type {Message, OrchestratorStatus} from './types/chat';

export default function App() {
    const [sessionId, setSessionId] = useState<string | null>(null);
    const [messages, setMessages] = useState<Message[]>([
        {id: '1', role: 'assistant', content: 'Привет! Я твой локальный ИИ. Мой оркестратор готов к работе.'}
    ]);
    const [isLoading, setIsLoading] = useState(false);
    const [status, setStatus] = useState<OrchestratorStatus>('initializing');

    const messagesEndRef = useRef<HTMLDivElement>(null);

    // Функция для создания абсолютно нового чистого чата
    const startNewChatSession = async () => {
        setIsLoading(true);
        setStatus('initializing');
        try {
            const data = await initOrchestratorSession();
            setSessionId(data.sessionId);
            setMessages([
                {id: '1', role: 'assistant', content: 'Привет! Я твой автономный ИИ-маркетолог AdBroker. Опишите ваш бизнес и какую рекламу мы запускаем?'}
            ]);
            setStatus('connected');
        } catch (error) {
            console.error('Failed to init session:', error);
            setStatus('error');
        } finally {
            setIsLoading(false);
        }
    };

    // При первой загрузке приложения принудительно запрашиваем UUID у бэкенда
    useEffect(() => {
        startNewChatSession();
    }, []);

    useEffect(() => {
        messagesEndRef.current?.scrollIntoView({behavior: 'smooth'});
    }, [messages, isLoading]);

    const handleSendMessage = async (text: string) => {
        if (isLoading || !sessionId) return;

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
            // Передаем зафиксированный sessionId аргументом в клиент стриминга
            await sendChatCompletionStream(sessionId, updatedMessages, (fullCleanText) => {
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
            {/* Пробрасываем функцию сброса и текущий UUID в Sidebar */}
            <Sidebar status={status} sessionId={sessionId} onNewChat={startNewChatSession}/>

            <div className="flex-1 flex flex-col h-full bg-gray-900">
                <div className="h-16 border-b border-gray-800 px-8 flex items-center justify-between bg-gray-900/50 backdrop-blur">
                    <div className="font-medium">
                        {status === 'initializing' ? 'Инициализация ядра...' : 'Локальная сессия ассистента'}
                    </div>
                </div>

                <div className="flex-1 overflow-y-auto p-8 space-y-6">
                    {messages.map((msg) => (
                        <ChatMessage key={msg.id} message={msg}/>
                    ))}
                    {isLoading && messages.length > 1 && <ChatLoader/>}
                    <div ref={messagesEndRef}/>
                </div>

                {/* Блокируем инпут, пока идет инициализация UUID сессии */}
                <ChatInput
                    onSendMessage={handleSendMessage}
                    isLoading={isLoading || status === 'initializing'}
                />
            </div>
        </div>
    );
}
