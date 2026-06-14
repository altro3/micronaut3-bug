import { Cpu } from 'lucide-react';
import { type OrchestratorStatus } from '../../types/chat';

interface SidebarProps {
    status: OrchestratorStatus;
}

export default function Sidebar({ status }: SidebarProps) {
    return (
        <div className="w-80 bg-gray-950 border-r border-gray-800 p-6 flex flex-col justify-between">
            <div>
                <h1 className="text-xl font-bold tracking-wider mb-2 flex items-center gap-2">
                    <Cpu className="text-indigo-400" /> AI Orchestrator
                </h1>
                <p className="text-xs text-gray-500 mb-6">Панель мониторинга локального ядра</p>

                <div className="space-y-4">
                    <div>
                        <label className="text-xs text-gray-400 block mb-1">Статус оркестратора</label>
                        <div className="flex items-center gap-2 text-sm bg-gray-900 p-2.5 rounded-lg border border-gray-800">
                            <span className={`h-2.5 w-2.5 rounded-full ${status === 'connected' ? 'bg-green-500 animate-pulse' : 'bg-red-500'}`} />
                            {status === 'connected' ? 'Оркестратор активен' : 'Нет связи (порт 8083)'}
                        </div>
                    </div>

                    <div>
                        <label className="text-xs text-gray-400 block mb-1">Протокол контекста</label>
                        <div className="text-xs bg-gray-900 p-2.5 rounded-lg border border-gray-800 text-indigo-300 font-mono">
                            OpenAI Compatible API
                        </div>
                    </div>
                </div>
            </div>

            <div className="text-xs text-gray-600 border-t border-gray-800 pt-4">
                Разработано для демонстрации работы агентов и локальных LLM.
            </div>
        </div>
    );
}
