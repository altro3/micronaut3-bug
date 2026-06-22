import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import { Bot, User } from 'lucide-react';
import type { Message } from '../../types/chat';

interface ChatMessageProps {
    message: Message;
}

export default function ChatMessage({ message }: ChatMessageProps) {
    const isUser = message.role === 'user';

    return (
        <div className={`flex gap-4 max-w-3xl ${isUser ? 'ml-auto flex-row-reverse' : 'mr-auto'}`}>
            <div className={`h-9 w-9 rounded-lg flex items-center justify-center shrink-0 ${isUser ? 'bg-indigo-600' : 'bg-emerald-600'}`}>
                {isUser ? <User size={18} /> : <Bot size={18} />}
            </div>

            {/* Тело сообщения */}
            {/* ИСПРАВЛЕНО: Убрали whitespace-pre-wrap, так как ReactMarkdown сам управляет переносами блоков */}
            <div className="space-y-2 max-w-full overflow-hidden">
                <div className={`p-4 rounded-xl text-sm leading-relaxed ${isUser ? 'bg-indigo-500/10 border border-indigo-500/20 text-indigo-100' : 'bg-gray-800 border border-gray-700 text-gray-200'}`}>
                    {isUser ? (
                        <span className="whitespace-pre-wrap">{message.content}</span>
                    ) : (
                        <ReactMarkdown
                            remarkPlugins={[remarkGfm]}
                            components={{
                                p: ({ ...props }) => <p className="mb-2 last:mb-0 text-gray-300 leading-relaxed" {...props} />,
                                strong: ({ ...props }) => <strong className="font-bold text-indigo-400" {...props} />,

                                ul: ({ ...props }) => <ul className="list-disc pl-5 mt-1 mb-2 space-y-0.5" {...props} />,
                                ol: ({ ...props }) => <ol className="list-decimal pl-5 mt-1 mb-2 space-y-0.5" {...props} />,

                                li: ({ ...props }) => <li className="text-gray-300 leading-relaxed" {...props} />,
                                code: ({ ...props }) => <code className="bg-gray-950 px-1.5 py-0.5 rounded text-amber-400 font-mono text-xs" {...props} />,

                                h1: ({ ...props }) => <h1 className="text-xl font-bold text-gray-100 mt-4 mb-1.5" {...props} />,
                                h2: ({ ...props }) => <h2 className="text-lg font-bold text-gray-100 mt-3.5 mb-1.5" {...props} />,
                                h3: ({ ...props }) => <h3 className="text-base font-semibold text-gray-100 mt-3 mb-1" {...props} />,

                                table: ({ ...props }) => (
                                    <div className="overflow-x-auto my-4 rounded-xl border border-gray-700 bg-gray-900/40">
                                        <table className="min-w-full border-collapse divide-y divide-gray-700" {...props} />
                                    </div>
                                ),
                                th: ({ ...props }) => (
                                    <th
                                        className="px-4 py-3 bg-gray-900 text-left text-xs font-bold text-indigo-400 uppercase tracking-wider border-r border-gray-700 last:border-r-0 whitespace-normal"
                                        {...props}
                                    />
                                ),
                                td: ({ ...props }) => (
                                    <td
                                        className="px-4 py-3 text-xs border-b border-r border-gray-800 last:border-r-0 text-gray-300 whitespace-normal leading-relaxed align-top"
                                        {...props}
                                    />
                                )
                            }}
                        >
                            {message.content}
                        </ReactMarkdown>
                    )}
                </div>
            </div>
        </div>
    );
}
