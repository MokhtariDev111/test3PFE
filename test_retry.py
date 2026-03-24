"""Test retry logic."""
import asyncio
from modules.retry_utils import retry_async, RetryExhausted

# Counter to track attempts
attempt_count = 0

async def flaky_function():
    """Fails twice, succeeds on third attempt."""
    global attempt_count
    attempt_count += 1
    
    if attempt_count < 3:
        raise ConnectionError(f"Simulated failure #{attempt_count}")
    
    return "Success!"

async def test_retry():
    global attempt_count
    attempt_count = 0
    
    result = await retry_async(
        flaky_function,
        max_attempts=3,
        base_delay=0.1,  # Fast for testing
    )
    
    print(f"Result: {result}")
    print(f"Total attempts: {attempt_count}")
    assert result == "Success!", "Should succeed on third attempt"
    assert attempt_count == 3, "Should have taken 3 attempts"
    print("✅ Retry logic works!")

async def test_exhaustion():
    """Test that RetryExhausted is raised after max attempts."""
    global attempt_count
    attempt_count = 0
    
    async def always_fails():
        global attempt_count
        attempt_count += 1
        raise ConnectionError("Always fails")
    
    try:
        await retry_async(always_fails, max_attempts=3, base_delay=0.1)
        assert False, "Should have raised RetryExhausted"
    except RetryExhausted as e:
        print(f"Correctly exhausted after {e.attempts} attempts")
        assert e.attempts == 3
        print("✅ Retry exhaustion works!")

# Run tests
asyncio.run(test_retry())
asyncio.run(test_exhaustion())
print("\n✅ All retry tests passed!")