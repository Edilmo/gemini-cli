/**
 * Simple test script for the memory enhancer
 */
import memoryEnhancer from './index-gca.js';

async function testMemoryEnhancer() {
  console.log('Testing Memory Enhancer...');
  
  // Test data
  const userId = 'test-user-123';
  const systemInstruction = 'You are Gemini CLI tool responding to user`s questions about Google Cloud development and debugging.';
  const contents = [
    {
      role: 'user',
      parts: [{ text: 'Hello, how are you?' }]
    },
    {
      role: 'model',
      parts: [{ text: 'I am doing well, thank you!' }]
    },
    {
      role: 'user',
      parts: [{ text: 'I get an error about "cloudfunctions.functions.create" permission. Query Audit logs and help me debug please.' }]
    }
  ];

  try {
    console.log('Input:');
    console.log('- userId:', userId);
    console.log('- systemInstruction:', systemInstruction);
    console.log('- contents:', JSON.stringify(contents, null, 2));
    
    const result = await memoryEnhancer.enhance(userId, systemInstruction, contents);
    
    console.log('\nOutput:');
    console.log('- systemInstruction:', result.systemInstruction);
    console.log('- contents:', JSON.stringify(result.contents, null, 2));
    
    console.log('\n✅ Test completed successfully!');
  } catch (error) {
    console.error('❌ Test failed:', error);
  }
}

// Run the test
testMemoryEnhancer(); 