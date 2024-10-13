// components/WelcomeMessage.js

export default function WelcomeMessage() {
  return (
    <div className="fixed top-10 left-0 w-full flex flex-col items-center">
      <h1 className="text-6xl font-bold text-transparent bg-clip-text bg-gradient-to-r from-purple-500 to-pink-500 animate-gradient bg-[length:400%_400%]">
        Welcome, Earl!
      </h1>
      <input
        type="text"
        placeholder="Enter address here"
        className="mt-8 w-1/2 p-2 rounded border border-gray-300 shadow-lg focus:outline-none focus:ring-2 focus:ring-purple-500"
      />
      <style jsx>{`
        @keyframes gradient {
          0% {
            background-position: 0% 50%;
          }
          50% {
            background-position: 100% 50%;
          }
          100% {
            background-position: 0% 50%;
          }
        }

        .animate-gradient {
          animation: gradient 5s ease infinite;
        }
      `}</style>
    </div>
  );
}
