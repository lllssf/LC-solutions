class Solution {
public:
    void reverseString(vector<char>& s) 
    {
        int from = 0;
        int to = s.size()-1;
        while(from < to)
        {
            swap(s[from++],s[to--]);
        }
    }
};
