#This is a comment.  Begin your comments with hashtags.

test = 5	                            #A variable.  (No declaration needed.)
primes = [2, 3, 5, 7, 11, 13, 17, 19]	#A list.
print primes[2]	                        #Prints 5.  Python is 0-indexed.
sumOfPrimes = 0
for prime in primes:                    #Loops through each item in primes.
    sumOfPrimes = sumOfPrimes + prime   #Indenting is mandatory - Python is sensitive to indents.
    if sumOfPrimes > 40:                #An if statement.
        print "The sum is currently greater than 40."
    else:
        print "The sum is currently at most 40."
        
i = 0
sumOfPrimes = 0
while sumOfPrimes < 40:                 #A while loop.  Stops when sumOfPrimes >= 40.
    sumOfPrimes += primes[i]            #a += b means a = a + b.
    i += 1
    if i >= len(primes):                #len(list) returns list length.
        break                           #break instantly exits a while loop or a for loop.
print "The last prime used was " + primes[i-1]

primes.append(23)                       #list.append(thing) sticks thing at the end of the list.
print primes[-1]                        #[-1] accesses the last item in the list, so prints 23.