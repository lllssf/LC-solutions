def findMedianSortedArrays(nums1,nums2):
    m,n = len(nums1),len(nums2)
    # Ensure j is non-negative
    if m > n:
        m,n,nums1,nums2 = n,m,nums2,nums1

    imin, imax = 0,m
    while imin<=imax:
        # Ensure len(left)==len(right)
        i = (imin+imax)/2
        j = (m+n+1)/2 - i
        # If i is too small
        if i < m  and nums1[i] < nums2[j-1]:
            imin += 1
        # If i is too big
        elif i > 0 and nums1[i-1] > nums2[j]:
            imax -= 1
        else:
            # i is perfect
            if i==0 : max_of_left = nums2[j-1]
            elif j == 0: max_of_left = nums1[i-1]
            else: max_of_left = max(nums1[i-1],nums2[j-1])
            # (m+n) is odd
            if (m+n) % 2 == 1: return max_of_left
            # (m+n) is even
            if i == m: min_of_right = nums2[j]
            elif j == n: min_of_right = nums1[i]
            else: min_of_right = min(nums1[i],nums2[j])

            return (max_of_left + min_of_right) / 2.0
