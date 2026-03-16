import React from 'react';
import { ArrowRight, ShieldCheck, Zap, BookOpen, Activity } from 'lucide-react';
import Link from 'next/link';

export default function HomePage() {
  return (
    <div className="space-y-12">
      {/* Hero Section */}
      <section className="relative h-[400px] rounded-3xl overflow-hidden bg-gradient-to-r from-primary to-blue-600 text-white p-12 flex flex-col justify-center">
        <div className="absolute top-0 right-0 w-1/2 h-full opacity-10 pointer-events-none">
           <svg viewBox="0 0 200 200" xmlns="http://www.w3.org/2000/svg" className="w-full h-full">
            <path fill="#FFFFFF" d="M44.7,-76.4C58.8,-69.2,71.8,-59.1,79.6,-45.8C87.4,-32.6,90,-16.3,88.5,-0.9C86.9,14.5,81.2,29.1,72.4,41.4C63.6,53.8,51.7,64,38.5,71.5C25.3,79,10.7,83.8,-3,89C-16.7,94.2,-33.4,99.8,-47.3,94.1C-61.2,88.4,-72.3,71.4,-80.4,54.7C-88.5,38.1,-93.6,21.7,-91.5,6.2C-89.4,-9.3,-80.1,-23.9,-70.6,-37.4C-61.1,-50.9,-51.4,-63.3,-38.8,-71.4C-26.2,-79.5,-10.7,-83.4,3,-88.7C16.8,-93.9,30.6,-83.5,44.7,-76.4Z" transform="translate(100 100)" />
          </svg>
        </div>
        
        <div className="relative z-10 max-w-2xl">
          <h2 className="text-4xl md:text-5xl font-extrabold mb-6 leading-tight">
            Advanced Airway Assessment <br /> Powered by AI
          </h2>
          <p className="text-lg text-blue-100 mb-10 leading-relaxed">
            Revolutionizing preoperative and emergency intubation risk assessment using Deep Learning and ResNet18 architecture.
          </p>
          <div className="flex flex-wrap gap-4">
            <Link 
              href="/predict" 
              className="px-8 py-4 bg-white text-primary font-bold rounded-2xl hover:bg-blue-50 transition-colors flex items-center gap-2 group"
            >
              Start Prediction
              <ArrowRight size={20} className="group-hover:translate-x-1 transition-transform" />
            </Link>
            <Link 
              href="/research" 
              className="px-8 py-4 bg-white/10 text-white font-bold rounded-2xl hover:bg-white/20 transition-colors flex items-center gap-2"
            >
              Read Paper
              <BookOpen size={20} />
            </Link>
          </div>
        </div>
      </section>

      {/* Quick Stats/Features */}
      <section className="grid grid-cols-1 md:grid-cols-3 gap-8">
        <FeatureCard 
          icon={<ShieldCheck className="text-blue-500" size={32} />}
          title="Enhanced Safety"
          description="Early recognition enables preparation with advanced airway tools and expert personnel."
        />
         <FeatureCard 
          icon={<Zap className="text-yellow-500" size={32} />}
          title="Instant Results"
          description="Get rapid predictions for 'Easy' vs 'Difficult' intubation with confidence scores."
        />
         <FeatureCard 
          icon={<Activity className="text-green-500" size={32} />}
          title="Deep Learning"
          description="Trained on standardized clinical photos using state-of-the-art neural networks."
        />
      </section>

      {/* Content Section */}
      <section className="grid grid-cols-1 lg:grid-cols-2 gap-12 items-center">
        <div className="space-y-6">
          <h3 className="text-3xl font-bold text-slate-800 dark:text-white">What is Intubation?</h3>
          <p className="text-slate-600 dark:text-slate-400 leading-relaxed">
            Intubation is a critical medical procedure to secure a patient's airway by inserting a tube through the mouth into the trachea, ensuring proper ventilation. It is essential for major surgical procedures under general anesthesia, emergency situations like trauma, and managing critically ill patients.
          </p>
          <div className="bg-blue-50 dark:bg-blue-900/20 p-6 rounded-2xl border border-blue-100 dark:border-blue-800">
            <h4 className="font-bold text-blue-800 dark:text-blue-300 mb-2">Predicting Difficulty</h4>
            <ul className="space-y-2 text-blue-700 dark:text-blue-400">
              <li className="flex gap-2">
                <span className="text-blue-500">•</span>
                <span>Reduces procedural delays and complications</span>
              </li>
              <li className="flex gap-2">
                <span className="text-blue-500">•</span>
                <span>Optimizes resource and expert team allocation</span>
              </li>
              <li className="flex gap-2">
                <span className="text-blue-500">•</span>
                <span>Improves clinical safety and efficiency</span>
              </li>
            </ul>
          </div>
        </div>
        
        <div className="bg-white dark:bg-slate-800 p-8 rounded-3xl shadow-xl shadow-slate-200/50 dark:shadow-none border border-slate-100 dark:border-slate-700">
          <h3 className="text-xl font-bold mb-6 flex items-center gap-2">
             <div className="w-1 h-6 bg-primary rounded-full" />
             How to use AirwayAI
          </h3>
          <div className="space-y-8">
            <Step number="01" title="Capture Photos" description="Take three clinical photos: Neutral face, Tongue extended, and Sniffing position (Head up)." />
            <Step number="02" title="Upload Images" description="Securely upload the images for the AI to analyze anatomical risk markers." />
            <Step number="03" title="Get Prediction" description="Receive an instant risk profile based on our trained ResNet18 model." />
          </div>
        </div>
      </section>
    </div>
  );
}

function FeatureCard({ icon, title, description }: { icon: React.ReactNode; title: string; description: string }) {
  return (
    <div className="bg-white dark:bg-slate-800 p-8 rounded-3xl shadow-sm border border-slate-100 dark:border-slate-700 hover:shadow-md transition-shadow">
      <div className="mb-4">{icon}</div>
      <h4 className="text-lg font-bold mb-2">{title}</h4>
      <p className="text-slate-500 dark:text-slate-400 text-sm leading-relaxed">{description}</p>
    </div>
  );
}

function Step({ number, title, description }: { number: string; title: string; description: string }) {
  return (
    <div className="flex gap-6">
      <span className="text-3xl font-black text-slate-100 dark:text-slate-700 select-none">{number}</span>
      <div>
        <h5 className="font-bold mb-1">{title}</h5>
        <p className="text-slate-500 dark:text-slate-400 text-sm leading-relaxed">{description}</p>
      </div>
    </div>
  );
}
