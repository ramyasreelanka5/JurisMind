import { useState, useRef } from 'react';

const useAudioRecorder = () => {
    const [isRecording, setIsRecording] = useState(false);
    const [audioBlob, setAudioBlob] = useState(null);
    const mediaRecorderRef = useRef(null);
    const audioChunksRef = useRef([]);

    const startRecording = async () => {
        if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
            alert('Your browser does not support audio recording.');
            return;
        }

        try {
            const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
            setIsRecording(true);
            setAudioBlob(null);
            audioChunksRef.current = [];

            mediaRecorderRef.current = new MediaRecorder(stream, { mimeType: 'audio/webm' });

            mediaRecorderRef.current.ondataavailable = (event) => {
                if (event.data.size > 0) {
                    audioChunksRef.current.push(event.data);
                }
            };

            mediaRecorderRef.current.onstop = () => {
                const blob = new Blob(audioChunksRef.current, { type: 'audio/webm' });
                setAudioBlob(blob);
                // Stop all tracks on the stream to release the microphone
                stream.getTracks().forEach(track => track.stop());
            };

            mediaRecorderRef.current.start();
        } catch (err) {
            console.error('Error starting recording:', err);
            alert('Could not start recording. Please grant microphone permission.');
        }
    };

    const stopRecording = () => {
        if (mediaRecorderRef.current && isRecording) {
            mediaRecorderRef.current.stop();
            setIsRecording(false);
        }
    };

    return { isRecording, startRecording, stopRecording, audioBlob };
};

export default useAudioRecorder;