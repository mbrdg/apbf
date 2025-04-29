use std::{
    borrow::Borrow,
    collections::HashMap,
    f64::consts::LOG2_E,
    hash::{BuildHasher, Hash, RandomState},
    marker::PhantomData,
    num::NonZeroUsize,
    time::{Duration, Instant},
};

#[derive(Debug, Clone)]
pub struct AgePartionedBloomFilter<T: ?Sized> {
    k: NonZeroUsize,
    l: NonZeroUsize,
    generation_size: NonZeroUsize,
    inserts: usize,
    refresh_interval: Duration,
    last_refresh: Instant,
    buf: Vec<u8>,
    slice_bits: usize,
    index: usize,
    hashers: (RandomState, RandomState),
    marker: PhantomData<T>,
}

impl<T> AgePartionedBloomFilter<T> {
    pub fn new(k: NonZeroUsize, l: NonZeroUsize, generation_size: NonZeroUsize) -> Self {
        Self::with_refresh(k, l, generation_size, Duration::ZERO)
    }

    pub fn with_refresh(
        k: NonZeroUsize,
        l: NonZeroUsize,
        generation_size: NonZeroUsize,
        refresh_interval: Duration,
    ) -> Self {
        let slice_capacity = k.get() * generation_size.get();
        let slice_bits = f64::ceil(LOG2_E * slice_capacity as f64) as usize;
        let buf_len = ((k.get() + l.get()) * slice_bits).next_multiple_of(8) / 8;

        Self {
            k,
            l,
            generation_size,
            inserts: 0,
            refresh_interval,
            last_refresh: Instant::now(),
            buf: vec![0u8; buf_len],
            slice_bits,
            index: 0,
            hashers: (RandomState::new(), RandomState::new()),
            marker: PhantomData,
        }
    }

    pub const fn slices(&self) -> usize {
        self.k.get() + self.l.get()
    }

    const fn generations(&self) -> usize {
        self.l.get() + 1
    }

    pub const fn capacity(&self) -> usize {
        self.generation_size.get() * self.generations()
    }

    fn refresh(&mut self) {
        let now = Instant::now();
        if self.refresh_interval.is_zero() || self.last_refresh + self.refresh_interval > now {
            return;
        }

        while self.last_refresh + self.refresh_interval < now {
            self.shift();
            self.last_refresh += self.refresh_interval;
        }
    }

    pub fn contains<Q>(&mut self, value: &Q) -> bool
    where
        T: Borrow<Q>,
        Q: Eq + Hash + ?Sized,
    {
        self.refresh();

        let h1 = self.hashers.0.hash_one(value) as usize;
        let h2 = self.hashers.1.hash_one(value) as usize;

        let mut matches = 0;
        for i in 0..self.slices() {
            let bit = (i * self.slice_bits) + h1.wrapping_add(i.wrapping_mul(h2)) % self.slice_bits;

            let (byte, off) = (bit / 8, bit % 8);
            if let Some(b) = self.buf.get(byte) {
                matches += usize::from((b >> off) & 1);
            }

            if matches == self.k.get() {
                return true;
            }
        }

        false
    }

    fn shift(&mut self) {
        self.inserts = 0;
        self.index = match self.index {
            0 => self.slices() - 1,
            _ => self.index - 1,
        };

        let lo = self.index * self.slice_bits;
        let hi = (self.index + 1) * self.slice_bits;

        for i in lo..hi {
            let (byte, off) = (i / 8, i % 8);
            if let Some(b) = self.buf.get_mut(byte) {
                *b &= !(1 << off)
            }
        }
    }

    pub fn insert<Q>(&mut self, value: &Q)
    where
        T: Borrow<Q>,
        Q: Eq + Hash + ?Sized,
    {
        if self.inserts == self.generation_size.get() {
            self.shift()
        }

        self.inserts += 1;

        let h1 = self.hashers.0.hash_one(value) as usize;
        let h2 = self.hashers.1.hash_one(value) as usize;

        for k in 0..self.k.get() {
            let i = (self.index + k) % self.slices();
            let bit = (i * self.slice_bits) + h1.wrapping_add(i.wrapping_mul(h2)) % self.slice_bits;

            let (byte, off) = (bit / 8, bit % 8);
            if let Some(b) = self.buf.get_mut(byte) {
                *b |= 1 << off;
            }
        }
    }

    pub fn fpr(k: NonZeroUsize, l: NonZeroUsize) -> f64 {
        let mut cache = HashMap::new();
        Self::fpr_rec(k.get(), l.get(), (0, 0), &mut cache)
    }

    fn fpr_rec(
        k: usize,
        l: usize,
        cache_key @ (a, i): (usize, usize),
        cache: &mut HashMap<(usize, usize), f64>,
    ) -> f64 {
        if a == k {
            return 1.0;
        } else if i > l + a {
            return 0.0;
        } else if let Some(v) = cache.get(&cache_key) {
            return *v;
        }

        let mut ri = 0.5;
        if i < k {
            ri = (i + 1) as f64 / (2 * k) as f64;
        }

        let value = ri * Self::fpr_rec(k, l, (a + 1, i + 1), cache)
            + (1.0 - ri) * Self::fpr_rec(k, l, (0, i + 1), cache);
        cache.insert(cache_key, value);

        value
    }
}

#[cfg(test)]
mod tests {
    use std::thread::sleep;

    use super::*;

    type Filter = AgePartionedBloomFilter<i32>;

    #[test]
    fn fpr() {
        let epsilon = 1e-6;
        let items = vec![
            (4, 3, 0.100586),
            (5, 7, 0.101603),
            (7, 5, 0.011232),
            (8, 8, 0.010244),
            (10, 7, 0.001211),
            (11, 9, 0.000918),
            (14, 11, 0.000099),
            (15, 15, 0.000100),
            (17, 13, 0.000011),
            (18, 16, 0.000009),
        ];

        for (k, l, expected) in items {
            let actual = Filter::fpr(NonZeroUsize::new(k).unwrap(), NonZeroUsize::new(l).unwrap());
            assert!(f64::abs(actual - expected) < epsilon);
        }
    }

    #[test]
    fn simple_query() {
        let mut f = AgePartionedBloomFilter::<i32>::new(
            NonZeroUsize::new(10).unwrap(),
            NonZeroUsize::new(7).unwrap(),
            NonZeroUsize::new(5).unwrap(),
        );

        assert!(!f.contains(&0), "filter should be empty");
        assert!(!f.contains(&1), "filter should be empty");

        f.insert(&0);

        assert!(f.contains(&0), "filter should contain 0");
        assert!(!f.contains(&1), "filter should not contain 1");
    }

    #[test]
    fn eviction_by_expiration() {
        let mut f = AgePartionedBloomFilter::<i32>::with_refresh(
            NonZeroUsize::new(3).unwrap(),
            NonZeroUsize::new(6).unwrap(),
            NonZeroUsize::new(10).unwrap(),
            Duration::from_millis(50),
        );

        f.insert(&0);
        assert!(f.contains(&0), "item should not be evicted after insertion");

        sleep(Duration::from_millis(50));
        assert!(f.contains(&0), "item should not be evicted by this time");

        sleep(Duration::from_millis(2 * f.generations() as u64 * 50));
        assert!(!f.contains(&0), "item should be evicted by this time")
    }

    #[test]
    fn eviction_by_shifting() {
        let mut f = AgePartionedBloomFilter::<i32>::new(
            NonZeroUsize::new(3).unwrap(),
            NonZeroUsize::new(3).unwrap(),
            NonZeroUsize::new(10).unwrap(),
        );

        f.insert(&0);
        assert!(f.contains(&0), "item should not be evicted after insertion");

        for item in 1..=f.capacity() {
            f.insert(&(item as i32));
        }

        assert!(!f.contains(&0), "item should be evicted by this time");
    }
}
